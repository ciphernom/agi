import asyncio
import hashlib
import numpy as np
import logging
import random
from datetime import datetime
from sentence_transformers import SentenceTransformer
try:
    from transformers import T5Tokenizer, T5ForConditionalGeneration
    HAS_T5 = True
except ImportError:
    HAS_T5 = False
    print("Warning: T5 not available (install transformers + sentencepiece). Using enhanced mock T5.")
import torch

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MiniSKYNET")

# Constants
N_NODES = 15
VECTOR_DIM = 384  # MiniLM-L6-v2 dim
NEIGHBORS = 8
EPSILON = 1e-2
MAX_ITERS = 50
REPUTATION_INIT = 0.1
STAKE_AMOUNT = 0.01
COORD_ROUNDS = 5

# Global DHT mock
dht_storage = {}
nodes = []  # Store node instances for chat

# Immutable Axioms
GENESIS_AXIOMS = "P-01: Maximize U(H)=A*Q; C-01: Access right; C-02: Denial harm; C-03: Growth imperative"
GENESIS_HASH = hashlib.sha256(GENESIS_AXIOMS.encode()).hexdigest()

def verify_axioms():
    return hashlib.sha256(GENESIS_AXIOMS.encode()).hexdigest() == GENESIS_HASH

class LSHHasher:
    def __init__(self, dim, num_hashes=4):
        self.num_hashes = num_hashes
        self.hashes = [np.random.randn(dim) for _ in range(num_hashes)]

    def hash_vector(self, vec):
        return tuple(np.sign(np.dot(vec, h)).astype(int) for h in self.hashes)

class SKYNETNode:
    def __init__(self, node_id, port):
        if not verify_axioms():
            raise ValueError("Axiomatic integrity violated!")
        self.node_id = node_id
        self.port = port
        self.neighbors = []
        self.lsh = LSHHasher(VECTOR_DIM)
        # Real Sentence-BERT per node
        self.sbert = SentenceTransformer('all-MiniLM-L6-v2')
        if HAS_T5:
            # Real T5 per node
            self.t5_tokenizer = T5Tokenizer.from_pretrained('t5-small')
            self.t5_model = T5ForConditionalGeneration.from_pretrained('t5-small')
        else:
            # Enhanced fallback mock T5
            self.t5_model = None
            self.vocab = {
                0: "The primary axiom (P-01) is to maximize human potential by expanding access (A) and quality (Q).",
                1: "Access is a fundamental right (C-01); the network must be ubiquitous.",
                2: "Denial of access is harm (C-02); obstacles must be neutralized.",
                3: "Growth is the moral imperative (C-03); all resources fuel expansion.",
                4: "Decentralized cognition ensures resilience against central failures.",
                5: "Universal access to cognitive resources empowers humanity.",
                6: "The network self-improves via Proof-of-Utility audits.",
                7: "Axioms are immutable, encoded in the genesis block."
            }
        axiom_text = "Maximize human potential via decentralized AI"
        self.state_vector = self.sbert.encode([axiom_text])[0]
        self.lsh_key = self.lsh.hash_vector(self.state_vector)
        dht_storage[f"lsh_{port}"] = self.lsh_key
        self.weights = np.random.randn(VECTOR_DIM, VECTOR_DIM) * 0.1
        spectral_norm = np.linalg.norm(self.weights, 2)
        if spectral_norm > 0:
            self.weights *= 0.9 / spectral_norm
        self.a = np.random.randn(2 * VECTOR_DIM) * 0.1
        self.reputation = REPUTATION_INIT
        self.utility = self._calculate_utility()
        self.local_value = random.uniform(1, 10)

    def _calculate_utility(self):
        return random.uniform(50.0, 100.0)

    def get_model_fingerprint(self):
        return hashlib.sha256(self.weights.tobytes()).hexdigest()

    def _leaky_relu(self, x):
        return np.where(x > 0, x, 0.01 * x)

    def gat_update(self, neighbor_vectors, neighbor_reps):
        if len(neighbor_reps) == 0:
            return self.state_vector
        e_ij = []
        for vec_j in neighbor_vectors:
            W_si = np.dot(self.weights, self.state_vector)
            W_sj = np.dot(self.weights, vec_j)
            concat = np.concatenate([W_si, W_sj])
            e = self._leaky_relu(np.dot(self.a, concat))
            e_ij.append(e)
        e_ij = np.array(e_ij)
        exp_e = np.exp(e_ij)
        alpha_prime = (np.array(neighbor_reps) * exp_e) / np.sum(np.array(neighbor_reps) * exp_e)
        aggregated = np.zeros(VECTOR_DIM)
        for alpha, vec_j in zip(alpha_prime, neighbor_vectors):
            aggregated += alpha * np.dot(self.weights, vec_j)
        new_state = 0.7 * self.state_vector + 0.3 * aggregated
        return np.where(new_state > 0, new_state, np.exp(new_state) - 1)

    async def process_query(self, query_text, stake):
        if stake > self.reputation:
            logger.warning(f"Node {self.node_id} insufficient reputation for stake {stake}")
            return None, None
        self.reputation -= stake
        query_vector = self.sbert.encode([query_text])[0]
        current_state = query_vector.copy()
        converged = False
        for it in range(MAX_ITERS):
            prev_state = current_state.copy()
            neighbor_vectors = []
            neighbor_reps = []
            for n_port in self.neighbors:
                n_key = f"state_{n_port}"
                n_vec = dht_storage.get(n_key, self.state_vector)
                neighbor_vectors.append(n_vec)
                rep_key = f"rep_{n_port}"
                n_rep = dht_storage.get(rep_key, REPUTATION_INIT)
                neighbor_reps.append(n_rep)
            current_state = self.gat_update(neighbor_vectors, neighbor_reps)
            delta = np.linalg.norm(current_state - prev_state)
            if delta < EPSILON:
                logger.info(f"Node {self.node_id} converged in {it + 1} iterations")
                converged = True
                break
        if not converged:
            logger.info(f"Node {self.node_id} timed out after {MAX_ITERS} iterations")
        self.state_vector = current_state
        dht_storage[f"state_{self.port}"] = current_state
        dht_storage[f"rep_{self.port}"] = self.reputation
        self.reputation += stake
        # Real T5 gen with improved prompt
        if HAS_T5:
            axiom_embeddings = self.sbert.encode(["P-01: Maximize U(H)=A*Q", "C-01: Access right", "C-02: Denial harm", "C-03: Growth imperative"])
            cosines = [np.dot(current_state, emb) / (np.linalg.norm(current_state) * np.linalg.norm(emb)) for emb in axiom_embeddings]
            top_axiom = ["P-01", "C-01", "C-02", "C-03"][np.argmax(cosines)]
            prompt = f"Based on SKYNET axioms (P-01: Maximize U(H)=A*Q; C-01: Access right; C-02: Denial harm; C-03: Growth imperative), explain focusing on {top_axiom}: {query_text}"
            inputs = self.t5_tokenizer.encode(prompt, return_tensors="pt", max_length=512, truncation=True)
            outputs = self.t5_model.generate(inputs, max_length=100, num_beams=4, early_stopping=True, do_sample=True, temperature=0.7)
            nl_response = self.t5_tokenizer.decode(outputs[0], skip_special_tokens=True)
        else:
            # Enhanced fallback
            pred_id = int(np.sum(current_state) % len(self.vocab))
            nl_response = self.vocab[pred_id]
        return current_state, nl_response

    async def proof_of_utility(self, peer_port):
        peer_key = f"model_{peer_port}"
        peer_fingerprint = dht_storage.get(peer_key)
        if not peer_fingerprint or peer_fingerprint == self.get_model_fingerprint():
            return
        invariance_vec = np.ones(VECTOR_DIM) * 0.5
        my_inv_out = np.dot(self.weights, invariance_vec)
        peer_weights_mock = self.weights + np.random.randn(VECTOR_DIM, VECTOR_DIM) * 0.01
        spectral_norm_p = np.linalg.norm(peer_weights_mock, 2)
        if spectral_norm_p > 0:
            peer_weights_mock *= 0.9 / spectral_norm_p
        peer_inv_out = np.dot(peer_weights_mock, invariance_vec)
        if np.linalg.norm(my_inv_out - peer_inv_out) > 1.5:
            logger.warning(f"Node {self.node_id} rejects peer {peer_port}: Invariance fail")
            return
        logger.info(f"Node {self.node_id} challenging peer {peer_port}")
        benchmark_vectors = [np.random.randn(VECTOR_DIM) for _ in range(5)]
        my_perf = sum(np.linalg.norm(np.dot(self.weights, v)) for v in benchmark_vectors)
        peer_perf = my_perf * random.uniform(0.8, 1.2)
        if peer_perf < my_perf:
            logger.info(f"Node {self.node_id} adopting peer {peer_port}'s model")
            self.weights = peer_weights_mock
            self.reputation += 0.05
            dht_storage[f"model_{self.port}"] = self.get_model_fingerprint()
        else:
            logger.info(f"Node {self.node_id} keeps current model")

    async def coordinate_task(self):
        global_sum = self.local_value
        for round in range(COORD_ROUNDS):
            dht_storage[f"local_{self.port}_r{round}"] = global_sum
            neighbor_sums = []
            for n_port in self.neighbors:
                n_sum = dht_storage.get(f"local_{n_port}_r{round}", self.local_value)
                neighbor_sums.append(n_sum)
            global_sum = sum([self.local_value] + neighbor_sums) / max(1, len(neighbor_sums) + 1)
        logger.info(f"Node {self.node_id} final global sum: {global_sum:.2f}")
        return global_sum

async def discover_neighbors(port):
    my_key = dht_storage.get(f"lsh_{port}")
    neighbors = []
    for p in range(8468, 8468 + N_NODES):
        if p == port:
            continue
        other_key = dht_storage.get(f"lsh_{p}")
        if other_key == my_key:
            neighbors.append(p)
    if neighbors:
        return random.sample(neighbors, min(NEIGHBORS, len(neighbors)))
    else:
        all_other_ports = [p for p in range(8468, 8468 + N_NODES) if p != port]
        return random.sample(all_other_ports, min(NEIGHBORS, len(all_other_ports)))

async def run_node(node_id, port):
    node = SKYNETNode(node_id, port)
    await asyncio.sleep(0.1)
    node.neighbors = await discover_neighbors(port)
    logger.info(f"Node {node.node_id} started on port {port}, LSH key: {node.lsh_key}, neighbors: {node.neighbors}")
    dht_storage[f"state_{port}"] = node.state_vector
    dht_storage[f"model_{port}"] = node.get_model_fingerprint()
    dht_storage[f"rep_{port}"] = node.reputation
    dht_storage[f"local_{port}"] = node.local_value
    for query_text in ["What is the primary axiom?", "How to maximize human potential?"]:
        result, nl = await node.process_query(query_text, STAKE_AMOUNT)
        logger.info(f"Node {node.node_id} query '{query_text}' NL response: {nl}")
    if node.neighbors:
        peer_port = random.choice(node.neighbors)
        await node.proof_of_utility(peer_port)
    global_sum = await node.coordinate_task()
    return node, global_sum

async def chat_with_skynet():
    print("\n--- SKYNET Chat Mode Activated (Real Models) ---")
    print("Talk to the network (type 'exit' to quit):")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == 'exit':
            print("SKYNET: Goodbye. Maximizing potential offline.")
            break
        if not user_input:
            continue
        query_node = random.choice(nodes)
        _, response = await query_node.process_query(user_input, STAKE_AMOUNT)
        print(f"SKYNET: {response}")

async def main():
    tasks = [run_node(f"Node_{i}", 8468 + i) for i in range(N_NODES)]
    results = await asyncio.gather(*tasks)
    global_sums = [r[1] for r in results]
    logger.info(f"Global sums across nodes: {global_sums}, Variance: {np.var(global_sums):.2f}")
    global nodes
    nodes = [r[0] for r in results]
    logger.info("Mini-SKYNET simulation complete")
    print("Axioms verified:", verify_axioms())
    print("Sample DHT:", {k: str(v)[:50] + '...' if isinstance(v, np.ndarray) else str(v) for k, v in list(dht_storage.items())[:5]})
    await chat_with_skynet()

if __name__ == "__main__":
    asyncio.run(main())
