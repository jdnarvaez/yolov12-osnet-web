import * as ort from 'onnxruntime-web/webgpu';

export async function createSession(
  modelUrl: string,
  providers: ort.InferenceSession.ExecutionProviderConfig[],
): Promise<ort.InferenceSession> {
  for (const provider of providers) {
    console.log(`Trying to load [${provider}] for inference from ${modelUrl}`);

    try {
      const s = await ort.InferenceSession.create(modelUrl, {
        executionProviders: [provider],
        graphOptimizationLevel: 'all',
        enableCpuMemArena: true,
        enableMemPattern: true,
      });

      console.log(`Using ${provider} for inference with model ${modelUrl}`);
      return s;
    } catch (err) {
      console.warn(`[worker] ${provider} session creation failed`, err);
    }
  }

  throw new Error(`Failed to create inference sessio for model ${modelUrl}`);
}
