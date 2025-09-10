import fastify, { FastifyRequest, FastifyReply } from 'fastify';
import cors from '@fastify/cors';
import * as tf from '@tensorflow/tfjs-node';
import * as fs from 'fs/promises';
import * as path from 'path';
import { createClient } from '@supabase/supabase-js';
import { findBestMoveNN } from './src/ai';
import type { Player } from './src/ai';

// --- Configuration ---
const MODEL_DIR = './model_gcp'; // Local directory on the GCP server
const MCTS_THINK_TIME = 15000; // 15 seconds
const PORT = 8080;
const MODEL_CHECK_INTERVAL_MS = 5 * 60 * 1000; // Check every 5 minutes

// --- Supabase Configuration ---
const SUPABASE_URL = 'https://xkwgfidiposftwwasdqs.supabase.co';
const SUPABASE_ANON_KEY = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Inhrd2dmaWRpcG9zZnR3d2FzZHFzIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTUwODM3NzMsImV4cCI6MjA3MDY1OTc3M30.-9n_26ga07dXFiFOShP78_p9cEcIKBxHBEYJ1A1gaiE';
const supabase = createClient(SUPABASE_URL, SUPABASE_ANON_KEY);

let model: tf.LayersModel | null = null;
let currentModelTimestamp: string | null = null;

// --- Model Loading and Auto-Update from Supabase ---

async function downloadAndLoadModel(): Promise<boolean> {
    console.log('[Model Syncer] Downloading latest model from Supabase...');
    try {
        await fs.mkdir(MODEL_DIR, { recursive: true });

        const { data: jsonBlob, error: jsonError } = await supabase.storage.from('models').download('gomoku_model/model.json');
        if (jsonError) throw new Error(`Failed to download model.json: ${jsonError.message}`);
        await fs.writeFile(path.join(MODEL_DIR, 'model.json'), Buffer.from(await jsonBlob.arrayBuffer()));

        const { data: weightsBlob, error: weightsError } = await supabase.storage.from('models').download('gomoku_model/weights.bin');
        if (weightsError) throw new Error(`Failed to download weights.bin: ${weightsError.message}`);
        await fs.writeFile(path.join(MODEL_DIR, 'weights.bin'), Buffer.from(await weightsBlob.arrayBuffer()));

        console.log('[Model Syncer] Model downloaded. Loading into memory...');
        model = await tf.loadLayersModel(`file://${path.resolve(MODEL_DIR)}/model.json`);
        console.log('[Model Syncer] New model loaded successfully!');
        return true;
    } catch (e) {
        console.error('[Model Syncer] Failed to download or load model:', e);
        return false;
    }
}

async function checkForNewModel() {
    console.log('[Model Syncer] Checking for new model version in Supabase...');
    try {
        const { data, error } = await supabase.storage.from('models').list('gomoku_model', {
            limit: 1,
            offset: 0,
            sortBy: { column: 'updated_at', order: 'desc' },
        });

        if (error) throw error;

        if (data && data.length > 0) {
            const latestVersionTimestamp = data[0].updated_at;
            if (!currentModelTimestamp || latestVersionTimestamp > currentModelTimestamp) {
                console.log(`[Model Syncer] New model version detected! (New: ${latestVersionTimestamp}, Current: ${currentModelTimestamp})`);
                const success = await downloadAndLoadModel();
                if (success) {
                    currentModelTimestamp = latestVersionTimestamp;
                }
            } else {
                console.log('[Model Syncer] No new model detected.');
            }
        }
    } catch (e) {
        console.error('[Model Syncer] Error checking for new model:', e);
    }
}

// --- Server Setup ---

const server = fastify({ logger: true });
server.register(cors, { origin: "*" });

interface GetMoveRequestBody {
    board: (Player | null)[][];
    player: Player;
    moves: any[];
}

server.post('/get-move', async (request: FastifyRequest<{ Body: GetMoveRequestBody }>, reply: FastifyReply) => {
    if (!model) {
        return reply.status(503).send({ error: 'AI model is not ready or still loading.' });
    }
    try {
        const { board, player, moves } = request.body;
        if (!board || !player || !moves) {
            return reply.status(400).send({ error: 'Missing or invalid request body' });
        }
        const { bestMove } = await findBestMoveNN(model, board, player, MCTS_THINK_TIME);
        return reply.send({ move: bestMove });
    } catch (e) {
        server.log.error(e);
        return reply.status(500).send({ error: 'An internal error occurred' });
    }
});

async function start() {
    try {
        await checkForNewModel(); // Initial model load
        setInterval(checkForNewModel, MODEL_CHECK_INTERVAL_MS); // Periodically check for updates
        await server.listen({ port: PORT, host: '0.0.0.0' });
    } catch (err) {
        server.log.error(err);
        process.exit(1);
    }
}

start();