import Foundation
import Metal

/// Prefill helpers for `KVPageStore`: scatter a chunk into (possibly
/// non-identity) pool slots, write Quest min/max for pages the chunk seals,
/// and stream sealed past pages through a flash-attention window when the
/// hot pool cannot hold the whole context.
enum PagedKVPrefill {
    static let windowPages = 128

    struct Scratch {
        let chunkTokens: Int
        let mState: MTLBuffer
        let dState: MTLBuffer
        let oState: MTLBuffer
        let stages: [MTLBuffer]
        let stagingTable: MTLBuffer
        let tailTable: MTLBuffer

        init(device: MTLDevice, config: ArchConfig, chunkTokens: Int,
             kPageBytes: Int) throws {
            self.chunkTokens = chunkTokens
            let rows = chunkTokens * config.numHeads
            let headDim = config.fullHeadDim
            guard let m = device.makeBuffer(length: rows * 4, options: .storageModeShared),
                  let d = device.makeBuffer(length: rows * 4, options: .storageModeShared),
                  let o = device.makeBuffer(length: rows * headDim * 4,
                                            options: .storageModeShared) else {
                throw KVPageStoreError.allocationFailed("blocked prefill state")
            }
            m.label = "kvpage.flash.m"
            d.label = "kvpage.flash.d"
            o.label = "kvpage.flash.o"
            self.mState = m
            self.dState = d
            self.oState = o

            var stages: [MTLBuffer] = []
            for i in 0..<2 {
                guard let stage = device.makeBuffer(
                    length: windowPages * 2 * kPageBytes,
                    options: .storageModeShared) else {
                    throw KVPageStoreError.allocationFailed("blocked prefill staging")
                }
                stage.label = "kvpage.flash.stage\(i)"
                stages.append(stage)
            }
            self.stages = stages

            var identity = (0..<windowPages).map { UInt32(2 * $0) }
            guard let table = device.makeBuffer(bytes: &identity,
                                                length: identity.count * 4,
                                                options: .storageModeShared),
                  let tail = device.makeBuffer(
                    length: (chunkTokens / KVPageGeometry.tokensPerPage + 2) * 4,
                    options: .storageModeShared) else {
                throw KVPageStoreError.allocationFailed("blocked prefill tables")
            }
            table.label = "kvpage.flash.stagingTable"
            tail.label = "kvpage.flash.tailTable"
            self.stagingTable = table
            self.tailTable = tail
        }
    }

    static func scatter(commandBuffer cb: MTLCommandBuffer,
                        paged: Qwen38PagedKVRuntime,
                        layerIndex: Int,
                        startPosition: Int,
                        tokenCount: Int,
                        keySource: MTLBuffer,
                        valueSource: MTLBuffer,
                        bytesPerToken: Int) throws {
        let pageTokens = KVPageGeometry.tokensPerPage
        let firstPage = startPosition / pageTokens
        let lastPage = (startPosition + tokenCount - 1) / pageTokens
        guard let blit = cb.makeBlitCommandEncoder() else {
            throw KVPageStoreError.ioFailed(operation: "paged KV scatter", errno: 0)
        }
        for page in firstPage...lastPage {
            let writeStart = max(page * pageTokens, startPosition)
            let writeEnd = min((page + 1) * pageTokens, startPosition + tokenCount)
            let count = writeEnd - writeStart
            guard count > 0 else { continue }
            let kDst = try paged.store.kSlot(layer: layerIndex, position: writeStart)
            let vDst = try paged.store.vSlot(layer: layerIndex, position: writeStart)
            let srcOffset = (writeStart - startPosition) * bytesPerToken
            blit.copy(from: keySource, sourceOffset: srcOffset,
                      to: kDst.buffer, destinationOffset: kDst.offset,
                      size: count * bytesPerToken)
            blit.copy(from: valueSource, sourceOffset: srcOffset,
                      to: vDst.buffer, destinationOffset: vDst.offset,
                      size: count * bytesPerToken)
        }
        blit.endEncoding()
    }

    static func encodeSealedPageMinMax(commandBuffer cb: MTLCommandBuffer,
                                       paged: Qwen38PagedKVRuntime,
                                       config: ArchConfig,
                                       layerIndex: Int,
                                       startPosition: Int,
                                       tokenCount: Int) throws {
        let pageTokens = KVPageGeometry.tokensPerPage
        let firstSealed = startPosition / pageTokens
        let sealedEnd = (startPosition + tokenCount) / pageTokens
        guard sealedEnd > firstSealed,
              let ordinal = paged.store.fullLayerOrdinal(forLayer: layerIndex) else { return }
        let g = paged.store.geometry
        for page in firstSealed..<sealedEnd {
            guard let slot = paged.store.residentSlot(layer: layerIndex, pageIndex: page) else {
                throw KVPageStoreError.pageNotSealed(layer: layerIndex, pageIndex: page)
            }
            paged.kernels.encodePageMinMax(
                commandBuffer: cb,
                kPool: paged.store.kPoolBuffer(layer: layerIndex),
                slot: UInt32(slot),
                validTokens: UInt32(pageTokens),
                metadata: paged.store.metadataBuffer,
                metadataOffset: g.metadataOffset(layerOrdinal: ordinal, pageIndex: page),
                numKVHeads: UInt32(config.numFullKVHeads),
                headDim: UInt32(config.fullHeadDim))
        }
    }

    static func runBlockedAttention(paged: Qwen38PagedKVRuntime,
                                    blocked: Scratch,
                                    config: ArchConfig,
                                    layerIndex: Int,
                                    queryCount t: Int,
                                    startPosition: Int,
                                    query: MTLBuffer,
                                    output: MTLBuffer,
                                    makeCommandBuffer: () throws -> MTLCommandBuffer,
                                    finish: (MTLCommandBuffer) throws -> Void) throws {
        let g = paged.store.geometry
        let pageTokens = KVPageGeometry.tokensPerPage
        let headDim = UInt32(config.fullHeadDim)
        let numQ = UInt32(config.numHeads)
        let numKV = UInt32(config.numFullKVHeads)
        let qStride = UInt32(config.numHeads * config.fullHeadDim)
        let rows = UInt32(t * config.numHeads)
        let scale = Float(config.attentionScale)

        let initCB = try makeCommandBuffer()
        paged.kernels.encodeFlashInit(commandBuffer: initCB,
                                      mState: blocked.mState,
                                      dState: blocked.dState,
                                      oState: blocked.oState,
                                      rows: rows, headDim: headDim)
        initCB.commit()

        let pastPages = startPosition / pageTokens
        paged.store.flushSpills()
        var stageCBs: [MTLCommandBuffer?] = [nil, nil]
        var page = 0
        var windowIndex = 0
        while page < pastPages {
            let count = min(windowPages, pastPages - page)
            let stageIndex = windowIndex % 2
            if let previous = stageCBs[stageIndex] {
                previous.waitUntilCompleted()
            }
            try paged.store.readSpilledSpan(layer: layerIndex,
                                            firstPage: page, pageCount: count,
                                            into: blocked.stages[stageIndex])
            let windowCB = try makeCommandBuffer()
            paged.kernels.encodeFlashUpdate(
                commandBuffer: windowCB,
                q: query,
                kPool: blocked.stages[stageIndex],
                vPool: blocked.stages[stageIndex], vPoolOffset: g.kPageBytes,
                pageTable: blocked.stagingTable,
                mState: blocked.mState, dState: blocked.dState, oState: blocked.oState,
                queryCount: UInt32(t),
                qStartPosition: UInt32(startPosition),
                headDim: headDim, numQHeads: numQ, numKVHeads: numKV,
                windowStartPosition: UInt32(page * pageTokens),
                windowTokens: UInt32(count * pageTokens),
                qStrideElements: qStride,
                scale: scale,
                causal: false)
            windowCB.commit()
            stageCBs[stageIndex] = windowCB
            page += count
            windowIndex += 1
        }

        let endPage = (startPosition + t - 1) / pageTokens
        var tailSlots: [UInt32] = []
        for tailPage in pastPages...endPage {
            guard let slot = paged.store.residentSlot(layer: layerIndex, pageIndex: tailPage) else {
                throw KVPageStoreError.pageNotSealed(layer: layerIndex, pageIndex: tailPage)
            }
            tailSlots.append(UInt32(slot))
        }
        precondition(tailSlots.count * 4 <= blocked.tailTable.length,
                     "tail window exceeds its table")
        tailSlots.withUnsafeBytes { raw in
            blocked.tailTable.contents().copyMemory(from: raw.baseAddress!,
                                                    byteCount: raw.count)
        }

        let tailCB = try makeCommandBuffer()
        paged.kernels.encodeFlashUpdate(
            commandBuffer: tailCB,
            q: query,
            kPool: paged.store.kPoolBuffer(layer: layerIndex),
            vPool: paged.store.vPoolBuffer(layer: layerIndex),
            pageTable: blocked.tailTable,
            mState: blocked.mState, dState: blocked.dState, oState: blocked.oState,
            queryCount: UInt32(t),
            qStartPosition: UInt32(startPosition),
            headDim: headDim, numQHeads: numQ, numKVHeads: numKV,
            windowStartPosition: UInt32(pastPages * pageTokens),
            windowTokens: UInt32(startPosition + t - pastPages * pageTokens),
            qStrideElements: qStride,
            scale: scale,
            causal: true)
        paged.kernels.encodeFlashFinalize(commandBuffer: tailCB,
                                          mState: blocked.mState,
                                          dState: blocked.dState,
                                          oState: blocked.oState,
                                          out: output,
                                          queryCount: UInt32(t),
                                          headDim: headDim,
                                          numQHeads: numQ,
                                          oStrideElements: qStride)
        try finish(tailCB)
    }
}
