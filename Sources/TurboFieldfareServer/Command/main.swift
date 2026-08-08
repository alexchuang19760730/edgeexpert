import Darwin
import Foundation
import TurboFieldfareServerCore

let arguments: ServerArguments
do {
    arguments = try ServerArguments.parse(Array(CommandLine.arguments.dropFirst()))
} catch ServerArgumentError.help {
    print(ServerArguments.usage)
    exit(0)
} catch {
    FileHandle.standardError.write(Data("error: \(error)\n\n\(ServerArguments.usage)\n".utf8))
    exit(2)
}

do {
    let signals = ServerTerminationSignals()
    let modelURL = URL(fileURLWithPath: arguments.model).standardizedFileURL
    let backend = try await ServerModelSession.load(
        modelDirectory: modelURL,
        maxContext: arguments.maxContext,
        expertStreamingMode: arguments.expertStreamingMode,
        pdServiceMode: arguments.pdServiceMode,
        promptCacheMode: arguments.promptCacheMode,
        promptCachePrimingMode: arguments.promptCachePrimingMode,
        stickyQuotaMode: arguments.stickyQuotaMode,
        diagnosticsMode: arguments.diagnosticsMode,
        prefillExpertReadMode: arguments.prefillExpertReadMode,
        prefillExpertLayerLocalReadaheadExperts: arguments.prefillExpertLayerLocalReadaheadExperts,
        prefillExpertBoundedCoalescedRunExperts: arguments.prefillExpertBoundedCoalescedRunExperts,
        prefillExpertBoundedParallelMissReadWorkers: arguments.prefillExpertBoundedParallelMissReadWorkers,
        prefillExpertTraceOutput: arguments.prefillExpertTraceOutput)
    let server = TurboFieldfareHTTPServer(
        modelID: arguments.modelID,
        queueLimit: arguments.queueLimit,
        backend: backend)
    _ = try await server.start(port: arguments.port)
    print("TurboFieldfareServer ready at http://127.0.0.1:\(arguments.port) model=\(arguments.modelID) context=\(arguments.maxContext) runtime_profile=\(arguments.runtimeProfile.rawValue) expert_streaming=\(arguments.expertStreamingMode.rawValue) pd_service=\(arguments.pdServiceMode.rawValue) prompt_cache=\(arguments.promptCacheMode.rawValue) prompt_cache_priming=\(arguments.promptCachePrimingMode.rawValue) sticky_quota=\(arguments.stickyQuotaMode.rawValue) diagnostics=\(arguments.diagnosticsMode.rawValue) prefill_bounded_run=\(arguments.prefillExpertBoundedCoalescedRunExperts) prefill_bounded_workers=\(arguments.prefillExpertBoundedParallelMissReadWorkers) prefill_trace_output=\(arguments.prefillExpertTraceOutput ?? "off")")

    _ = await signals.wait()
    try await server.shutdown()
    await signals.cancel()
} catch {
    FileHandle.standardError.write(Data("error: \(error)\n".utf8))
    exit(1)
}
