/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  // The ASR worker runs @huggingface/transformers (onnxruntime-web) in the browser.
  // Keep it out of the server graph so nothing tries to bundle ORT for Node.
  serverExternalPackages: ['@huggingface/transformers', 'onnxruntime-web'],
  webpack: (config, { isServer }) => {
    if (!isServer) {
      // transformers.js ships .wasm/.mjs side-files for the ONNX runtime.
      config.module.rules.push({
        test: /\.wasm$/,
        type: 'asset/resource',
      })
    }
    return config
  },
}

export default nextConfig
