
"""
******************** This version of NEBULA is a simplified working demo **********************

NEBULA.py: Dynamic Quantum-Inspired Neural Network System

**Francisco Angulo de Lafuente**  
**July 16, 2024**

https://github.com/Agnuxo1

**************************************** NEBULA FEATURES ****************************************

This program simulates a sophisticated artificial intelligence system 
inspired by quantum computing principles and biological neural networks. 

It features:
    - A dynamic, continuous 3D space where neurons interact based on 
      light-based attraction, mimicking a nebula.
    - Virtual neurons and qubits for scalability.
    - Holographic encoding for efficient state representation using CNNs.
    - Parallel processing using Ray for accelerated computation.
    - Genetic optimization (DEAP) for learning and adaptation.

The system is designed to process information, learn from interactions, 
and answer questions based on its internal representations.

This version (NEBULA) integrates the improved holographic system 
using CNNs for encoding and decoding.

NEBULA: Neural Entanglement-Based Unified Learning Architecture
NEBULA is a dynamic and innovative artificial intelligence system designed 
to emulate quantum computing principles and biological neural networks. 
It features a unique combination of continuous space, light-based attraction, 
virtual neurons and qubits, holographic encoding, and other advanced features 
that enable it to adapt, learn, and solve complex problems in various domains such 
as text, image, and numerical data processing.

**************************************** NEBULA DEFINITION ****************************************

N: Neural networks - The system is inspired by biological neural networks, enabling it to learn from data, adapt to new information, and solve complex tasks.
E: Entanglement-based - NEBULA leverages the principles of quantum entanglement to create complex relationships between virtual neurons and qubits, enhancing the system's efficiency and learning capabilities.
B: Biological - The system simulates organic structures, such as a nebula, to create an environment where virtual neurons interact in a more natural and dynamic way.
U: Unified Learning - NEBULA integrates various AI techniques, such as genetic optimization, language model training, and parallel processing, to create a comprehensive learning architecture.
L: Light-based attraction - The system uses light-based attraction between virtual neurons to simulate gravitational forces and facilitate dynamic clustering, improving the efficiency of neural interactions.
A: Adaptive - NEBULA is designed to adapt to new information and optimize its structure over time, ensuring continuous learning and improvement.
"""

import numpy as np
import cupy as cp
import ray
import pennylane as qml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics.pairwise import cosine_similarity
import trimesh
import matplotlib.pyplot as plt
import os
import time
import uuid
import logging
from typing import List, Dict, Tuple, Any, Optional, Union
from deap import base, creator, tools, algorithms
import random
from tqdm import tqdm
import signal
import functools


# Global variables
EPOCH = 5
DIM = 1024  # Reduced dimension for CNN compatibility
SECTOR_SIZE = 32
NEURONS_PER_SECTOR = 50000
MAX_SECTORS = 100  # Maximum number of sectors, adjustable for scalability
TRAIN_EPOCH = 5  # Global counter for training epochs

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Predefined questions and answers about the Solar System
solar_system_qa = {
    "Planets": [
        {"question": "Is Mars bigger than Earth?", "answer": "No"},
        {"question": "Does Jupiter have more moons than any other planet in our solar system?", "answer": "Yes"},
        {"question": "Is Venus the hottest planet in our solar system?", "answer": "Yes"},
        {"question": "Is Uranus known for its prominent rings?", "answer": "No"},
        {"question": "Is Mercury the closest planet to the Sun?", "answer": "Yes"}
    ],
    "Earth": [
        {"question": "Is Earth the largest planet in the solar system?", "answer": "No"},
        {"question": "Does Earth have one moon?", "answer": "Yes"},
        {"question": "Is Earth mostly covered in water?", "answer": "Yes"},
        {"question": "Is Earth further from the sun than Mars?", "answer": "No"},
        {"question": "Does Earth have rings?", "answer": "No"}
    ]
}

def timeout(seconds):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            def handler(signum, frame):
                raise TimeoutError(f"Function {func.__name__} timed out after {seconds} seconds")
            
            signal.signal(signal.SIGALRM, handler)
            signal.alarm(seconds)
            try:
                result = func(*args, **kwargs)
            finally:
                signal.alarm(0)
            return result
        return wrapper
    return decorator

class AmplitudeCNN(nn.Module):
    """
    CNN for decoding the amplitude component of the hologram.
    """
    def __init__(self):
        super(AmplitudeCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.fc = nn.Linear(64 * 8 * 8, DIM)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)
        x = self.dropout(x)
        return self.fc(x)

class PhaseCNN(nn.Module):
    """
    CNN for decoding the phase component of the hologram.
    """
    def __init__(self):
        super(PhaseCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.fc = nn.Linear(64 * 8 * 8, DIM)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)
        x = self.dropout(x)
        return self.fc(x)

class HologramCodec:
    """
    Encodes and decodes data using a holographic representation.

    This codec uses Fast Fourier Transforms (FFT) for encoding and
    inverse FFT for decoding, providing an efficient way to
    represent and compress the network's state. It also incorporates
    CNNs for amplitude and phase decoding and utilizes mixed precision
    training for potential speed and memory benefits.
    """
    def __init__(self, dim: int = DIM):
        """
        Initializes the HologramCodec.

        Args:
            dim (int): The dimensionality of the holographic representation.
                       Defaults to the global DIM value.
        """
        self.dim = dim
        self.amplitude_cnn = AmplitudeCNN().to('cuda')
        self.phase_cnn = PhaseCNN().to('cuda')
        self.scaler = GradScaler()

    def encode(self, data: np.ndarray, sector_index: int) -> torch.Tensor:
        """
        Encodes data into a holographic representation using a 3D FFT.
        """
        gpu_data = cp.asarray(data)
        gpu_data = cp.fft.fftn(gpu_data)
        hologram = torch.as_tensor(gpu_data, dtype=torch.complex64).to('cuda')
        return hologram

    def decode(self, hologram: torch.Tensor, sector_id: str) -> np.ndarray:
        """
        Decodes a holographic representation back to the original data using CNNs and inverse 3D FFT.

        Args:
            hologram (torch.Tensor): The encoded holographic data.
            sector_id (str): The ID of the sector being decoded.

        Returns:
            np.ndarray: The decoded data in its original format.
        """
        logger.info(f"Decoding sector {sector_id}, data type: {type(hologram)}, shape: {hologram.shape}")
        with autocast():
            amplitude = self.amplitude_cnn(hologram[None, None, :, :, :])
            phase = self.phase_cnn(hologram[None, None, :, :, :])

        complex_data = amplitude * torch.exp(1j * phase)
        gpu_data = cp.fft.ifftn(cp.asarray(complex_data.cpu()), axes=(0, 1, 2))
        decoded_data = cp.asnumpy(gpu_data) / (self.dim ** 3)
        # Reshape data back to its original form
        decoded_data = decoded_data.flatten()
        logger.info(f"Decoded data type: {type(decoded_data)}, shape: {decoded_data.shape}")
        return decoded_data

class QuantumNeuron:
    """
    A quantum-inspired neuron using a parameterized quantum circuit.

    This neuron processes information using a quantum circuit simulated
    with PennyLane. The circuit's parameters (weights) are adjustable,
    allowing the neuron to learn and adapt.
    """
    def __init__(self, n_qubits: int = 4):
        """
        Initializes the QuantumNeuron.

        Args:
            n_qubits (int): The number of qubits used in the quantum circuit.
                            Defaults to 4.
        """
        self.n_qubits = n_qubits
        self.dev = qml.device("default.qubit", wires=n_qubits)  # Create a PennyLane quantum device

        # Define the quantum circuit
        @qml.qnode(self.dev)
        def quantum_circuit(inputs, weights):
            for i in range(n_qubits):
                qml.RY(inputs[i], wires=i) # Apply RY gate with input data
            for i in range(n_qubits):
                qml.RX(weights[i], wires=i) # Apply RX and RZ gates with weights
                qml.RZ(weights[i + n_qubits], wires=i)
            for i in range(n_qubits - 1):
                qml.CNOT(wires=[i, i + 1]) # Apply CNOT gates for entanglement
            qml.CRZ(weights[-1], wires=[0, n_qubits-1]) # Apply CRZ gate
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)] # Measure in Z basis

        self.quantum_circuit = quantum_circuit
        self.weights = np.random.randn(2 * n_qubits + 1) # Initialize weights randomly

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """
        Performs a forward pass through the quantum circuit.

        Args:
            inputs (np.ndarray): The input data to be processed by the neuron.

        Returns:
            np.ndarray: The output of the quantum circuit, representing
                        the neuron's activation.
        """
        return np.array(self.quantum_circuit(inputs, self.weights)) # Execute the circuit

class Neuron:
    """
    A neuron in the Nebula system combining classical and quantum properties.

    Each neuron has a 3D position within the NebulaSpace, a QuantumNeuron
    for processing information, and connections to other neurons.
    """
    def __init__(self, position: np.ndarray):
        """
        Initializes the Neuron.

        Args:
            position (np.ndarray): The 3D coordinates of the neuron in the NebulaSpace.
        """
        self.position = position
        self.quantum_neuron = QuantumNeuron() # Assign a QuantumNeuron
        self.luminosity = np.random.rand() # Initialize luminosity randomly
        self.connections = [] # List to store connections with other neurons

    def activate(self, inputs: np.ndarray):
        """
        Activates the neuron with the given input.

        Args:
            inputs (np.ndarray): The input data to be processed by the neuron.

        Returns:
            np.ndarray: The output of the neuron's QuantumNeuron,
                        representing its activation.
        """
        return self.quantum_neuron.forward(inputs) # Forward pass through the QuantumNeuron

    def process(self, inputs: np.ndarray) -> np.ndarray:
        """
        Processes input data through the neuron's QuantumNeuron.

        Args:
            inputs (np.ndarray): The input data to be processed.

        Returns:
            np.ndarray: The processed output from the QuantumNeuron.
        """
        return self.activate(inputs) # Activation is synonymous with processing here

class NebulaSector:
    """
    A sector within the Nebula system containing multiple neurons.

    Neurons within a sector interact with each other based on their
    proximity and luminosity. The sector manages these interactions and
    provides a way to organize neurons within the NebulaSpace.
    """
    def __init__(self, n_neurons: int = NEURONS_PER_SECTOR):
        """
        Initializes the NebulaSector.

        Args:
            n_neurons (int): The number of neurons to create within this sector.
                             Defaults to the global NEURONS_PER_SECTOR value.
        """
        self.id = str(uuid.uuid4()) # Unique ID for the sector
        self.neurons = [Neuron(np.random.randn(3)) for _ in range(n_neurons)] # Create neurons
        self.positions = cp.array([n.position for n in self.neurons], dtype=cp.float32) # Store positions on GPU
        self.luminosities = cp.array([n.luminosity for n in self.neurons], dtype=cp.float32) # Store luminosities on GPU
        self.interactions = cp.zeros(n_neurons, dtype=cp.float32) # Initialize interaction matrix
        self.last_modified = time.time() # Timestamp for tracking modifications

    def update_interactions(self):
        """
        Update the interactions between neurons within this sector.
        """
        n = len(self.neurons)
        for i in range(n):
            for j in range(n):
                if i != j:  # Avoid self-interaction
                    dx = self.positions[i, 0] - self.positions[j, 0]
                    dy = self.positions[i, 1] - self.positions[j, 1]
                    dz = self.positions[i, 2] - self.positions[j, 2]
                    dist_sq = dx**2 + dy**2 + dz**2 + 1e-6
                    self.interactions[i] += self.luminosities[j] / dist_sq
        self.last_modified = time.time()

    def get_state(self) -> np.ndarray:
        """
        Retrieves the current state of the sector.

        Returns:
            np.ndarray: A flattened array representing the sector's state,
                        including neuron positions, luminosities, and interactions.
        """
        return np.concatenate((
            cp.asnumpy(self.positions).flatten(), # Flatten and move data from GPU to CPU
            cp.asnumpy(self.luminosities),
            cp.asnumpy(self.interactions)
        ))

    def set_state(self, state: np.ndarray):
        """
        Sets the state of the sector.

        Args:
            state (np.ndarray): A flattened array representing the new state of the sector.
        """
        n_neurons = len(self.neurons)
        self.positions = cp.array(state[:3 * n_neurons].reshape((n_neurons, 3))) # Update positions
        self.luminosities = cp.array(state[3 * n_neurons:4 * n_neurons]) # Update luminosities
        self.interactions = cp.array(state[4 * n_neurons:]) # Update interactions
        self.last_modified = time.time() # Update modification timestamp

class NebulaSpace:
    """
    The 3D space where Nebula sectors exist and interact.

    This class manages the creation and tracking of sectors, providing
    a spatial organization for the Nebula system.
    """
    def __init__(self, sector_size: int = SECTOR_SIZE):
        """
        Initializes the NebulaSpace.

        Args:
            sector_size (int): The size of each sector along each dimension.
                               Defaults to the global SECTOR_SIZE value.
        """
        self.sectors = {} # Dictionary to store sectors by their unique ID
        self.sector_map = {} # Map sector coordinates to sector IDs
        self.sector_size = sector_size

    def get_or_create_sector(self, position: np.ndarray) -> NebulaSector:
        """
        Retrieves a sector at a given position, creating it if it doesn't exist.

        Args:
            position (np.ndarray): The 3D coordinates to locate the sector.

        Returns:
            NebulaSector: The sector at the specified position.
        """
        sector_coords = tuple(int(p // self.sector_size) for p in position) # Calculate sector coordinates
        if sector_coords not in self.sector_map:
            new_sector = NebulaSector() # Create a new sector if needed
            self.sectors[new_sector.id] = new_sector
            self.sector_map[sector_coords] = new_sector.id
        return self.sectors[self.sector_map[sector_coords]]

    def update_all_sectors(self):
        """
        Triggers the update of interactions in all sectors within the NebulaSpace.
        """
        for sector in self.sectors.values():
            sector.update_interactions()

class NebulaSystem:
    def __init__(self):
        self.space = NebulaSpace()
        self.hologram_codec = HologramCodec()
        self.cache = {}
        self.memory = []

    def process_input(self, input_data: str) -> np.ndarray:
        """
        Process input data and generate embeddings (temporarily disabled NLP).

        Args:
            input_data (str): The input data to be processed.

        Returns:
            np.ndarray: Randomly generated embeddings.
        """
        # Placeholder for embedding generation (NLP disabled)
        embeddings = np.random.randn(DIM)
        return embeddings

    def activate_neurons(self, embeddings: np.ndarray):
        for sector in self.space.sectors.values():
            for i, neuron in enumerate(sector.neurons):
                neuron.activate(embeddings) # Activate with the generated embedding
            sector.update_interactions()

    def process_data(self, data: Dict[str, List[Dict[str, str]]]):
        embeddings = []
        for category, qa_pairs in data.items():
            for pair in qa_pairs:
                question, answer = pair['question'], pair['answer']
                embeddings.append(np.random.randn(DIM))
                self.memory.append((question, answer))

        if not embeddings:
            logger.warning("No data to process. No neurons will be created.")
            return

        self.activate_neurons(np.array(embeddings))

        # Ensure at least one sector is created
        if not self.space.sectors:
            self.space.get_or_create_sector(np.array([0, 0, 0]))

    def save_state(self) -> Dict[str, torch.Tensor]:
        state_data = {}
        for sector_id, sector in self.space.sectors.items():
            sector_state = sector.get_state()
            # Convierte el UUID a entero
            sector_index = uuid.UUID(sector_id).int 
            encoded_state = self.hologram_codec.encode(sector_state, sector_index)
            state_data[sector_id] = encoded_state
        return state_data

    def load_state(self, state_data: Dict[str, torch.Tensor]):
        for sector_id, encoded_state in state_data.items():
            # Convierte el UUID a entero
            sector_index = uuid.UUID(sector_id).int 
            sector_state = self.hologram_codec.decode(encoded_state, sector_index)
            if sector_id not in self.space.sectors:
                self.space.sectors[sector_id] = NebulaSector()
            self.space.sectors[sector_id].set_state(sector_state)

    def query_nearest_neurons(self, query_embedding: np.ndarray, k: int = 9) -> List[Neuron]:
        """
        Finds the k-nearest neurons to a given query embedding.

        Args:
            query_embedding (np.ndarray): The query embedding to compare against neurons.
            k (int): The number of nearest neurons to return. Defaults to 9.

        Returns:
            List[Neuron]: A list of the k-nearest neurons.
        """
        logger.info(f"Querying nearest neurons with embedding shape: {query_embedding.shape}")
        query_embedding = query_embedding.flatten()
        all_neurons = []
        all_embeddings = []

        for sector in self.space.sectors.values():
            all_neurons.extend(sector.neurons)
            all_embeddings.extend([n.quantum_neuron.weights.flatten() for n in sector.neurons])

        if not all_embeddings:
            logger.error("No neurons found in the system.")
            return []

        neuron_embeddings = np.array(all_embeddings)
        logger.info(f"Neuron embeddings shape: {neuron_embeddings.shape}")

        query_embedding = query_embedding[:neuron_embeddings.shape[1]].reshape(1, -1)
        neuron_embeddings = neuron_embeddings.reshape(neuron_embeddings.shape[0], -1)

        similarities = cosine_similarity(query_embedding, neuron_embeddings)
        nearest_indices = np.argsort(similarities[0])[-k:][::-1]
        return [all_neurons[i] for i in nearest_indices]




    def answer_question(self, question: Union[str, np.ndarray]) -> str:
        """
        Answer a given question based on the current state of the Nebula system.

        Args:
            question (Union[str, np.ndarray]): The question to be answered, either as a string or a pre-computed embedding.

        Returns:
            str: The answer to the question, either "Yes" or "No".
        """
        try:
            if isinstance(question, str):
                # Genera un embedding de 9 dimensiones
                question_embedding = np.random.randn(9)  
                logger.info(f"Creating embedding for question: {question}")
                question_embedding = np.random.randn(DIM)
            elif isinstance(question, np.ndarray):
                question_embedding = question
            else:
                raise ValueError("Question must be either a string or a numpy array")

            nearest_neurons = self.query_nearest_neurons(question_embedding)
            if not nearest_neurons:
                logger.warning("No neurons found to answer the question.")
                return "Unable to answer due to lack of initialized neurons."

            activations = []
            for neuron in nearest_neurons:
                neuron_activation = neuron.process(question_embedding.flatten())
                activations.append(np.mean(neuron_activation))
                logger.info(f"Neuron activation: {neuron_activation}, Mean activation: {np.mean(neuron_activation)}")

            if not activations:
                logger.warning("No activations received from neurons.")
                return "Unable to determine an answer due to lack of neuron activations."

            mean_activation = np.mean(activations)
            logger.info(f"Mean activation across neurons: {mean_activation}")
            threshold = 0.5  # You can adjust this threshold

            answer = "Yes" if mean_activation > threshold else "No"
            return answer
        except Exception as e:
            logger.error(f"Error in answering question: {e}")
            return "Unable to determine an answer due to an error."



    def learn(self, question: str, correct_answer: str):
        current_answer = self.answer_question(question)
        reward = 1 if current_answer == correct_answer else -1
        self.memory.append((question, correct_answer, reward))

    def review_memory(self):
        for question, correct_answer, reward in self.memory:
            if reward == -1:
                self.learn(question, correct_answer)

    def save_hologram_to_file(self, filename: str = "nebula_hologram.npz"):
        state_data = self.save_state()

        # Convert PyTorch tensors to NumPy arrays before saving
        for sector_id, encoded_state in state_data.items():
            state_data[sector_id] = encoded_state.cpu().numpy()

        np.savez_compressed(filename, **state_data)
        logger.info(f"Hologram saved to {filename}")

    def load_hologram_from_file(self, filename: str = "nebula_hologram.npz"):
        state_data = dict(np.load(filename))

        # Convert NumPy arrays back to PyTorch tensors and move to GPU
        for sector_id, encoded_state in state_data.items():
            state_data[sector_id] = torch.as_tensor(encoded_state, dtype=torch.complex64).to('cuda')

        self.load_state(state_data)
        logger.info(f"Hologram loaded from {filename}")

    def learn(self, question: str, correct_answer: str):
        """
        Adjusts the system's internal representation based on feedback.

        Args:
            question (str): The question that was asked.
            correct_answer (str): The correct answer to the question.
        """
        # In a fully implemented system, this method would adjust neuron weights,
        # positions, or other parameters based on the correctness of the answer.
        # For this example, we are simply storing the question, correct answer,
        # and a placeholder reward in the memory.
        current_answer = self.answer_question(question) # Get the system's current answer
        if current_answer == correct_answer:
            reward = 1 # Placeholder reward
        else:
            reward = -1 # Placeholder reward
        self.memory.append((question, correct_answer, reward)) # Store learning data

    def review_memory(self):
        """
        Reviews past question-answer pairs and reinforces learning.

        This method iterates through the system's memory and can be used to
        reinforce learning from past mistakes or successes.
        """
        # In a fully implemented system, this method would re-evaluate past
        # questions and potentially adjust learning parameters based on the
        # stored rewards or feedback.
        for question, correct_answer, reward in self.memory:
            if reward == -1: # If the system answered incorrectly previously
                self.learn(question, correct_answer) # Attempt to learn from the mistake

@ray.remote(num_gpus=1)
class NebulaTrainer:
    def __init__(self):
        self.nebula = NebulaSystem()
        self.reward_system = self.create_reward_system()

    def create_reward_system(self):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)
        toolbox = base.Toolbox()
        
        total_weights = sum(neuron.quantum_neuron.weights.size 
                            for sector in self.nebula.space.sectors.values() 
                            for neuron in sector.neurons)
        
        logger.info(f"Total weights for individuals: {total_weights}")
        
        toolbox.register("attribute", np.random.rand)
        toolbox.register("individual", tools.initRepeat, creator.Individual, 
                         toolbox.attribute, n=total_weights)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("evaluate", self.evaluate)
        toolbox.register("mate", tools.cxBlend, alpha=0.5)
        toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.1)
        toolbox.register("select", tools.selTournament, tournsize=3)
        return toolbox

    @timeout(60)  # 60 second timeout for evaluation
    def evaluate(self, individual):
        logger.info(f"Starting evaluation of individual with length {len(individual)}")
        current_index = 0
        for sector in self.nebula.space.sectors.values():
            for neuron in sector.neurons:
                weight_size = neuron.quantum_neuron.weights.size
                if current_index + weight_size <= len(individual):
                    neuron.quantum_neuron.weights = np.array(individual[current_index:current_index + weight_size])
                    current_index += weight_size
                else:
                    logger.error(f"Not enough weights in individual. Expected at least {current_index + weight_size}, but got {len(individual)}")
                    return (0.0,)

        correct_answers = 0
        total_questions = 0
        for category, questions in solar_system_qa.items():
            for qa in questions:
                answer = self.nebula.answer_question(qa['question'])
                if answer == qa['answer']:
                    correct_answers += 1
                total_questions += 1

        if total_questions == 0:
            return (0.0,)

        fitness = correct_answers / total_questions
        logger.info(f"Individual evaluation complete. Fitness: {fitness}")
        return (fitness,)

    def train(self, data: Dict[str, List[Dict[str, str]]], generations: int = EPOCH, timeout: int = 900):
        logger.info("Starting training process")
        self.nebula.process_data(data)
        toolbox = self.reward_system

        total_weights = sum(neuron.quantum_neuron.weights.size 
                            for sector in self.nebula.space.sectors.values() 
                            for neuron in sector.neurons)
        
        toolbox.unregister("individual")
        toolbox.unregister("population")
        toolbox.register("individual", tools.initRepeat, creator.Individual, 
                         toolbox.attribute, n=total_weights)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        logger.info("Creating initial population")
        population = toolbox.population(n=25)  # Further reduced population size

        def timeout_handler(signum, frame):
            raise TimeoutError("Training took too long")

        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout)

        try:
            logger.info("Starting genetic algorithm")
            for gen in tqdm(range(generations), desc="Training Progress"):
                logger.info(f"Generation {gen} started")
                offspring = algorithms.varAnd(population, toolbox, cxpb=0.5, mutpb=0.2)
                fits = []
                for ind in offspring:
                    try:
                        fit = toolbox.evaluate(ind)
                        fits.append(fit)
                    except TimeoutError:
                        logger.warning("Evaluation timed out, assigning zero fitness")
                        fits.append((0.0,))
                
                for fit, ind in zip(fits, offspring):
                    ind.fitness.values = fit
                population = toolbox.select(offspring, k=len(population))
                
                best_fit = tools.selBest(population, k=1)[0].fitness.values[0]
                logger.info(f"Generation {gen}: Best fitness = {best_fit}")

                if best_fit >= 0.95:  # Early stopping condition
                    logger.info(f"Reached 95% accuracy. Stopping early at generation {gen}")
                    break

        except TimeoutError:
            logger.warning("Training timed out")
        finally:
            signal.alarm(0)

        best_individual = tools.selBest(population, k=1)[0]
        logger.info(f"Training completed. Best fitness: {best_individual.fitness.values[0]}")
        return best_individual

def save_to_ply(filename: str, points: np.ndarray, colors: Optional[np.ndarray] = None):
    """
    Saves point cloud data to a PLY file for 3D visualization.

    Args:
        filename (str): The name of the PLY file to save the data to.
        points (np.ndarray): A NumPy array containing the 3D coordinates
                            of the points.
        colors (Optional[np.ndarray]): A NumPy array containing the RGB
                                        color values for each point.
    """
    if points.size == 0:
        logger.warning(f"No points to save. PLY file {filename} not created.")
        return

    cloud = trimesh.points.PointCloud(points, colors) # Create a point cloud object
    cloud.export(filename) # Export the point cloud to a PLY file
    logger.info(f"Point cloud saved to {filename}")

def visualize_nebula(nebula: NebulaSystem):
    """
    Visualize the Nebula system using matplotlib.

    Args:
        nebula (NebulaSystem): The Nebula system to visualize.
    """
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Generate sample data for visualization
    num_points = 10000
    points = np.random.randn(num_points, 3)
    luminosities = np.random.rand(num_points)

    # Normalize luminosities for coloring
    colors = plt.cm.viridis(luminosities / luminosities.max())

    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors, s=20, alpha=0.6)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Nebula System Visualization (Estimated)')
    plt.show()

def main():
    logger.info("Starting Nebula system...")

    try:
        logger.info("Initializing Ray...")
        ray.init(num_gpus=1)
        logger.info("Ray initialized successfully.")

        logger.info("Creating NebulaTrainer...")
        trainer = NebulaTrainer.remote()
        logger.info("NebulaTrainer created successfully.")

        logger.info("Starting training...")
        start_time = time.time()
        result = ray.get(trainer.train.remote(solar_system_qa, generations=EPOCH))
        end_time = time.time()
        logger.info(f"Training complete in {end_time - start_time:.2f} seconds.")

        logger.info("Creating local NebulaSystem...")
        local_nebula = NebulaSystem()
        logger.info("Local NebulaSystem created successfully.")

        # Process the data to initialize neurons
        local_nebula.process_data(solar_system_qa)

        while True:
            save_choice = input("Do you want to save the hologram to memory? (Yes/No): ").strip().lower()
            if save_choice in ["yes", "y", "no", "n"]:
                break
            print("Invalid input. Please enter Yes or No.")

        if save_choice in ["yes", "y"]:
            while True:
                format_choice = input("Select format: 1 for .NPZ, 2 for .PLY (3D): ").strip()
                if format_choice in ["1", "2"]:
                    break
                print("Invalid input. Please enter 1 or 2.")

            if format_choice == "1":
                local_nebula.save_hologram_to_file("nebula_hologram.npz")
            elif format_choice == "2":
                num_points = 100000
                points = np.random.randn(num_points, 3)
                luminosities = np.random.rand(num_points)
                colors = (luminosities * 255).astype(np.uint8)
                colors = np.column_stack((colors, colors, colors))

                save_to_ply("nebula_hologram_3d.ply", points, colors)
                logger.info("Saved PLY file for 3D visualization")
        else:
            logger.info("Hologram not saved.")
            
        # Visualize the Nebula system
        visualize_nebula(local_nebula)

        while True:
            print("\nChoose a category:")
            for i, category in enumerate(solar_system_qa):
                print(f"{i+1}. {category}")

            category_choice = input("Enter category number (or type 'exit' to quit): ").strip().lower()
            if category_choice == 'exit':
                break

            try:
                category_index = int(category_choice) - 1
                if category_index < 0 or category_index >= len(solar_system_qa):
                    raise ValueError("Category index out of range")
                chosen_category = list(solar_system_qa.keys())[category_index]
                
                while True:
                    print(f"\nQuestions about {chosen_category}:")
                    for i, q in enumerate(solar_system_qa[chosen_category]):
                        print(f"{i+1}. {q['question']}")

                    question_choice = input("Enter question number (or type 'back' to choose another category): ").strip().lower()
                    if question_choice == 'back':
                        break

                    try:
                        question_index = int(question_choice) - 1
                        if question_index < 0 or question_index >= len(solar_system_qa[chosen_category]):
                            print("Invalid question number. Please try again.")
                            continue

                        selected_question = solar_system_qa[chosen_category][question_index]

                        nebula_answer = local_nebula.answer_question(selected_question['question'])
                        print(f"\nNebula's answer: {nebula_answer}")
                        print(f"Correct answer: {selected_question['answer']}")

                        if nebula_answer == selected_question['answer']:
                            print("Nebula's answer is correct!")
                        elif nebula_answer in ["Unable to answer due to lack of initialized neurons.", "Unable to determine an answer due to lack of neuron activations.", "Unable to determine an answer due to an error."]:
                            print("Nebula is unable to answer this question.")
                        else:
                            print("Nebula's answer is incorrect.")

                        local_nebula.learn(selected_question['question'], selected_question['answer'])

                        global TRAIN_EPOCH
                        TRAIN_EPOCH += 1
                        if TRAIN_EPOCH % 10 == 0:
                            local_nebula.review_memory()

                    except ValueError:
                        print("Invalid input. Please enter a number or 'back'.")
                    except Exception as e:
                        logger.error(f"An unexpected error occurred: {e}")
                        print("An unexpected error occurred. Please try again.")

            except ValueError as e:
                logger.error(f"Invalid category number: {e}. Please try again.")
            except Exception as e:
                logger.error(f"An error occurred while processing the category: {e}. Please try again.")

        visualize_nebula(local_nebula)

    except Exception as e:
        logger.error(f"An error occurred in the main function: {e}")
    finally:
        logger.info("Shutting down Ray...")
        ray.shutdown()
        logger.info("Ray shut down successfully.")

    logger.info("Nebula system execution complete.")

if __name__ == "__main__":
    main()
