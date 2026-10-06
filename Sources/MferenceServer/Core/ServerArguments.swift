import Foundation

public struct ServerArguments: Equatable, Sendable {
    public let model: String
    public let port: Int
    public let bindMode: ServerBindMode
    /// Explicit --model-id value; nil defers to the loaded model's family
    /// default for the loaded model family.
    public let modelIDOverride: String?
    public var modelID: String { modelIDOverride ?? "gemma-4-26b-a4b-it" }
    public let maxContext: Int
    public let queueLimit: Int
    public let promptCacheMode: ServerPromptCacheMode
    /// Explicit expert-cache slot count. nil keeps the RAM auto profile
    /// (96 slots on ≥24 GiB). 16 is the ~1.45 GB Qwen working set.
    public let expertCacheSlots: Int?
    /// off | on | auto. auto turns paged KV on above 16K so 32K+ does not
    /// allocate a linear full-attn cache.
    public let kvPaged: ServerKVPagedMode
    /// Resident pages per full-attention layer. nil uses 48 when paged is on
    /// (the ~60 MB Qwen 3.6 hot pool) or the RAM auto budget otherwise.
    public let kvPoolPages: Int?
    public let kvTopKPages: Int?

    public static let usage = """
    usage: MferenceServer --model <completed .gturbo directory> [options]

      --model <dir>          Required model directory.
      --port <1...65535>     Listening port (default 8080).
      --bind <mode>          loopback or tailnet (default loopback). tailnet
                             binds only the machine's Tailscale IPv4 address
                             and fails when Tailscale is unavailable.
      --model-id <id>        API model identifier (default derived from the
                             installed model: gemma-4-26b-a4b-it,
                             qwen3.6-35b-a3b, deepseek-v4-flash-2bit-dq,
                             inkling-small-4bit, or maple-preview-2bit-mlx).
      --max-context <tokens> 4096, 8192, 16384, 32768, 65536, or 128000 (default 16384).
      --queue-limit <count>  Maximum queued requests (default 4).
      --prompt-cache-mode <off|single-prefix>
                             Prompt KV reuse mode (default single-prefix).
      --expert-cache-slots <8|16|24|32|64|96|128>
                             Routed-expert LFU slots. Default is RAM auto
                             (96 on ≥24 GiB). Pass 16 for the ~1.45 GB profile.
      --kv-paged <off|on|auto>
                             Paged full-attention KV with SSD spill. auto is
                             on when --max-context is greater than 16384.
      --kv-pool-pages <n>    Hot pages per full-attention layer (default 48
                             when paged is on — ~60 MB for Qwen 3.6).
      --kv-top-k-pages <n>   Quest top-k pages (default 36 when paged is on).
      --help                 Show this help.
    """

    public static func parse(_ input: [String]) throws -> ServerArguments {
        var model: String?
        var port = 8080
        var bindMode = ServerBindMode.loopback
        var modelIDOverride: String?
        var maxContext = 16_384
        var queueLimit = 4
        var promptCacheMode: ServerPromptCacheMode = .singlePrefix
        var expertCacheSlots: Int?
        var kvPaged: ServerKVPagedMode = .auto
        var kvPoolPages: Int?
        var kvTopKPages: Int?
        var index = 0
        while index < input.count {
            let flag = input[index]
            if flag == "--help" || flag == "-h" { throw ServerArgumentError.help }
            guard index + 1 < input.count else {
                throw ServerArgumentError.invalid("\(flag) requires a value")
            }
            let value = input[index + 1]
            index += 2
            switch flag {
            case "--model":
                model = value
            case "--port":
                guard let parsed = Int(value), (1...65_535).contains(parsed) else {
                    throw ServerArgumentError.invalid("--port must be between 1 and 65535")
                }
                port = parsed
            case "--bind":
                guard let parsed = ServerBindMode(rawValue: value) else {
                    throw ServerArgumentError.invalid("--bind must be loopback or tailnet")
                }
                bindMode = parsed
            case "--model-id":
                guard !value.isEmpty else {
                    throw ServerArgumentError.invalid("--model-id must not be empty")
                }
                modelIDOverride = value
            case "--max-context":
                guard let parsed = Int(value),
                      [4_096, 8_192, 16_384, 32_768, 65_536, 128_000].contains(parsed) else {
                    throw ServerArgumentError.invalid("--max-context is not supported")
                }
                maxContext = parsed
            case "--queue-limit":
                guard let parsed = Int(value), parsed > 0 else {
                    throw ServerArgumentError.invalid("--queue-limit must be positive")
                }
                queueLimit = parsed
            case "--prompt-cache-mode":
                guard let parsed = ServerPromptCacheMode(rawValue: value) else {
                    throw ServerArgumentError.invalid(
                        "--prompt-cache-mode must be off or single-prefix")
                }
                promptCacheMode = parsed
            case "--expert-cache-slots":
                guard let parsed = Int(value),
                      [8, 16, 24, 32, 64, 96, 128].contains(parsed) else {
                    throw ServerArgumentError.invalid(
                        "--expert-cache-slots must be 8, 16, 24, 32, 64, 96, or 128")
                }
                expertCacheSlots = parsed
            case "--kv-paged":
                guard let parsed = ServerKVPagedMode(rawValue: value) else {
                    throw ServerArgumentError.invalid("--kv-paged must be off, on, or auto")
                }
                kvPaged = parsed
            case "--kv-pool-pages":
                guard let parsed = Int(value), parsed >= 8, parsed <= 2048 else {
                    throw ServerArgumentError.invalid("--kv-pool-pages must be 8...2048")
                }
                kvPoolPages = parsed
            case "--kv-top-k-pages":
                guard let parsed = Int(value), parsed >= 1, parsed <= 256 else {
                    throw ServerArgumentError.invalid("--kv-top-k-pages must be 1...256")
                }
                kvTopKPages = parsed
            default:
                throw ServerArgumentError.invalid("unknown flag: \(flag)")
            }
        }
        guard let model else { throw ServerArgumentError.invalid("--model is required") }
        return ServerArguments(model: model,
                               port: port,
                               bindMode: bindMode,
                               modelIDOverride: modelIDOverride,
                               maxContext: maxContext,
                               queueLimit: queueLimit,
                               promptCacheMode: promptCacheMode,
                               expertCacheSlots: expertCacheSlots,
                               kvPaged: kvPaged,
                               kvPoolPages: kvPoolPages,
                               kvTopKPages: kvTopKPages)
    }
}

/// Interface the server listens on. Resolution fails rather than widening:
/// there is no path from `.tailnet` to a wildcard or LAN address.
public enum ServerBindMode: String, Equatable, Sendable {
    case loopback
    case tailnet

    /// Resolves the listening address. `.tailnet` asks the Tailscale CLI for
    /// this machine's IPv4 address; `tailnetAddresses` is the seam tests use to
    /// supply that output without a Tailscale install.
    public func host(
        tailnetAddresses: () throws -> String = ServerBindMode.tailscaleIPv4Output
    ) throws -> String {
        switch self {
        case .loopback: "127.0.0.1"
        case .tailnet: try Self.tailnetHost(from: tailnetAddresses())
        }
    }

    /// Accepts exactly one Tailscale IPv4 address. Empty, ambiguous, IPv6-only,
    /// malformed, and off-range output all fail; none of them fall back.
    static func tailnetHost(from output: String) throws -> String {
        let fields = output.split(whereSeparator: \.isWhitespace)
        guard let first = fields.first else {
            throw ServerArgumentError.invalid(
                "tailscale reported no IPv4 address; ensure Tailscale is running and connected")
        }
        guard fields.count == 1 else {
            throw ServerArgumentError.invalid(
                "tailscale reported \(fields.count) IPv4 addresses; refusing to guess which to bind")
        }
        guard let address = tailscaleIPv4(String(first)) else {
            throw ServerArgumentError.invalid(
                "tailscale reported \"\(first.prefix(64))\", which is not a Tailnet IPv4 address")
        }
        return address
    }

    /// Returns the address unchanged when it is a dotted-quad IPv4 inside
    /// 100.64.0.0/10, the range Tailscale allocates from. Restricting to that
    /// range keeps a wildcard, loopback, or LAN address from ever being bound.
    private static func tailscaleIPv4(_ text: String) -> String? {
        let parts = text.split(separator: ".", omittingEmptySubsequences: false)
        guard parts.count == 4 else { return nil }
        var octets: [UInt8] = []
        octets.reserveCapacity(4)
        for part in parts {
            guard (1...3).contains(part.count),
                  part.allSatisfy({ $0.isASCII && $0.isNumber }),
                  part.count == 1 || part.first != "0",
                  let octet = UInt8(part) else { return nil }
            octets.append(octet)
        }
        guard octets[0] == 100, (64...127).contains(octets[1]) else { return nil }
        return text
    }

    /// Raw stdout of `tailscale ip -4`. Spawned directly with no shell, so
    /// nothing is interpolated into a command line.
    public static func tailscaleIPv4Output() throws -> String {
        let process = Process()
        let output = Pipe()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/env")
        process.arguments = ["tailscale", "ip", "-4"]
        process.standardOutput = output
        process.standardError = FileHandle.nullDevice
        do {
            try process.run()
        } catch {
            throw ServerArgumentError.invalid(
                "could not run tailscale; install its CLI and keep it on PATH")
        }
        let data = output.fileHandleForReading.readDataToEndOfFile()
        process.waitUntilExit()
        guard process.terminationStatus == 0 else {
            throw ServerArgumentError.invalid(
                "tailscale ip -4 exited with status \(process.terminationStatus); ensure the tailscale CLI is on PATH and Tailscale is running")
        }
        guard let text = String(data: data, encoding: .utf8) else {
            throw ServerArgumentError.invalid("tailscale ip -4 returned non-UTF-8 output")
        }
        return text
    }
}

public enum ServerKVPagedMode: String, Equatable, Sendable {
    case off
    case on
    case auto

    /// Collapses `auto` using the context length. `.on` / `.off` stay as-is.
    func resolved(maxContext: Int) -> ServerKVPagedMode {
        switch self {
        case .off, .on: return self
        case .auto: return maxContext > 16_384 ? .on : .off
        }
    }
}

public enum ServerArgumentError: Error, Equatable, CustomStringConvertible {
    case help
    case invalid(String)

    public var description: String {
        switch self {
        case .help: "help"
        case .invalid(let message): message
        }
    }
}
