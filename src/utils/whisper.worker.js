import { pipeline, env } from "@xenova/transformers";
// import { pipeline, env } from "@huggingface/transformers";
import { MessageTypes } from "./presets";
env.allowLocalModels = false;

class MyTranscriptionPipeline {
    static task = "automatic-speech-recognition";
    // static model = "openai/whisper-tiny.en";
     static model = "Xenova/whisper-base.en";
    //static model = "Xenova/whisper-small";
    static instance = null;

    static async getInstance(progress_callback = null) {
        if (this.instance === null) {
            this.instance = await pipeline(this.task, this.model, {
                progress_callback
            })
        }
        return this.instance;
    }
}

self.addEventListener("message", async (event) => {
    const { type, audio } = event.data;
    if (type === MessageTypes.INFERENCE_REQUEST) {
        await transcribe(audio)
    }
})

    /**
     * Transcribes an audio buffer using the Whisper ASR model.
     *
     * The provided audio buffer is processed in chunks, and the generated
     * text is sent back to the main thread as it is generated.
     *
     * The function takes an audio buffer as input, and returns a Promise that
     * resolves when the transcription is complete.
     *
     * @param {AudioBuffer} audio - The audio buffer to transcribe.
     *
     * @returns {Promise<void>}
     */
async function transcribe(audio) {
    sendLoadingMessage("loading");

    let pipeline;

    try {
        pipeline = await MyTranscriptionPipeline.getInstance(load_model_callback);
    } catch (err) {
        console.log(err)
        console.log(err.message)
    }

    sendLoadingMessage("success");

    const stride_length_s = 5;
    const generationTracker = new GenerationTracker(pipeline, stride_length_s);
    await pipeline(audio, {
        top_k: 0,
        do_sample: false,
        chunk_length: 30,
        stride_length_s: stride_length_s,
        return_timestamps: true,
        callback_function: generationTracker.callbackFunction.bind(generationTracker),
        chunk_callback: generationTracker.chunkCallback.bind(generationTracker),
    })
    generationTracker.sendFinalResult();
}

async function load_model_callback(data) {
    const { status } = data;
    if (status === "progress") {
        const { file, progress, loaded, total } = data;
        sendDownloadingMessage(file, progress, loaded, total);
    }
}

function sendLoadingMessage(status) {
    self.postMessage({
        type: MessageTypes.LOADING,
        status
    })
}

async function sendDownloadingMessage(file, progress, loaded, total) {
    self.postMessage({
        type: MessageTypes.DOWNLOADING,
        file,
        progress,
        loaded,
        total,
    })
}

class GenerationTracker {
/**
 * Constructs a GenerationTracker instance.
 *
 * @param {Object} pipeline - The transcription pipeline used for processing audio data.
 * @param {number} stride_length_s - The length of the stride in seconds for processing audio chunks.
 *
 * Initializes the pipeline, stride length, and other properties like chunk, time_precision,
 * processed_chunks, and callbackFunctionCounter. The time_precision is calculated based on
 * the feature extractor and model configuration within the pipeline.
 */
    constructor(pipeline, stride_length_s) {
        console.log("pipeline:", pipeline);
        console.log("pipeline.model:", pipeline.model);
        console.log("pipeline.model.config:", pipeline.model && pipeline.model.config);
        this.pipeline = pipeline;
        this.stride_length_s = stride_length_s;
        this.chunks = [];
        this.time_precision = pipeline?.processor.feature_extractor.config.chunk_length / pipeline.model.config.max_source_prositions;
        this.processed_chunks = [];
        this.callbackFunctionCounter = 0;
    }

    sendFinalResult() {
        self.postMessage({ type: MessageTypes.INFERENCE_DONE });
    }

    /**
     * Processes a set of beams returned by the transcription pipeline and sends a partial recognition result.
     *
     * The callback function is called by the transcription pipeline for every new set of beams generated during
     * the audio processing. This function is responsible for extracting the speech recognition output from the
     * best beam and sending a partial recognition result to the main thread.
     *
     * @param {Array} beams - A list of beams returned by the transcription pipeline.
     */
    callbackFunction(beams) {
        this.callbackFunctionCounter += 1;
        if (this.callbackFunctionCounter % 10 !== 0) {
            return
        }

        // Get the speech-recognition output
        const bestBeam = beams[0];
        let text = this.pipeline.tokenizer.decode(bestBeam.output_token_ids, {
            skip_special_tokens: true
        })

        const result = {
            text,
            start: this.getlastChunkTimestamp(),
            end: undefined,
        }

        createPartialResultMessage(result);
    }

    chunkCallback(data) {
        this.chunks.push(data)
        // eslint-disable-next-line no-unused-vars
        const [ text, {chunks}] = this.pipeline.tokenizer._decode_asr(
            this.chunks, 
            {
                time_precision: this.time_precision,
                return_timestamps: true,
                force_full_sequence: false,
            }
        );

        this.processed_chunks = chunks.map((chunk, index) => {
            return this.processChunk(chunk, index)
        });

        createResultMessage(
            this.processed_chunks, false, this.getlastChunkTimestamp()
        );
    };

    getlastChunkTimestamp(){
        if (this.processed_chunks.length=== 0) {
            return 0;
        }
    }

    processChunk(chunk, index) {
        const {text, timestamp} = chunk;
        const [start, end] = timestamp;

        return {
            index,
            text: `${text.trim()}`,
            start: Math.round(start),
            end: Math.round(end) || Math.round(start + 0.9 * this.stride_length_s )
        }
        
    }
};

function createResultMessage(results, isDone, completedUntilTimestamp) {
    self.postMessage({
        type: MessageTypes.RESULT,
        results,
        isDone,
        completedUntilTimestamp
    });
};

function createPartialResultMessage(result) {
    self.postMessage({
        type: MessageTypes.RESULT_PARTIAL,
        result
    });
};

