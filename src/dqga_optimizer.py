"""
Module 5: Discrete Quantum Genetic Algorithm (DQGA) Optimizer
============================================================
Performs hyperparameter optimization inspired by quantum computing 
principles (qubit probability amplitudes and rotation gates). 
This module aims to automatically find the optimal values for 
critical parameters (e.g., h, rho) across the PnP-ADMM and SCSA filters.

Author: FizyGame
Date: 2026
"""

from __future__ import annotations

import logging
from typing import Callable, Tuple, List, Optional

import numpy as np

logger = logging.getLogger("dqga_optimizer")

class QuantumChromosome:
    """
    A data structure representing a String of Qubits (Quantum Chromosome).
    Each gene (qubit) has complex probability amplitudes α and β.
    Mathematical rule: |α|^2 + |β|^2 = 1.
    """
    def __init__(self, n_genes: int) -> None:
        self.n_genes = n_genes
        # Initially, all are in equiprobable superposition: alpha = 1/sqrt(2), beta = 1/sqrt(2)
        base_val = 1.0 / np.sqrt(2.0)
        self.alpha = np.full(n_genes, base_val, dtype=np.float64)
        self.beta = np.full(n_genes, base_val, dtype=np.float64)
        
        # The classical bit string resulting from probability measurement (collapse)
        self.binary_string: Optional[np.ndarray] = None
        self.fitness: float = -np.inf

    def measure(self) -> np.ndarray:
        """
        Collapses the qubits down to classical bits (0 or 1) (Wave-function collapse).
        Probability of observing 1: |β|^2
        Probability of observing 0: |α|^2
        """
        probabilities_of_1 = self.beta ** 2
        # Probability simulation using np.random.rand
        random_vals = np.random.rand(self.n_genes)
        self.binary_string = (random_vals <= probabilities_of_1).astype(np.int8)
        return self.binary_string

    def _normalize(self) -> None:
        """Ensures the |α|² + |β|² = 1 constraint holds true following floating point errors."""
        norm = np.sqrt(self.alpha**2 + self.beta**2)
        # Division-by-zero isolation
        norm[norm == 0] = 1.0
        self.alpha /= norm
        self.beta /= norm

class QuantumGate:
    """Contains logic for Quantum Gate operations."""
    
    @staticmethod
    def rotate(chromosome: QuantumChromosome, best_chromosome: QuantumChromosome, theta: float = 0.05 * np.pi) -> None:
        """
        Quantum Rotation Gate that evolves the chromosome's probability distribution
        towards the most fit individual in the population.
        
        Args:
            chromosome: Target chromosome to update
            best_chromosome: Chromosome boasting the highest fitness in the population
            theta: Rotation angle magnitude (in radians)
        """
        # The genes undergoing rotation depends on classical bit observations
        if chromosome.binary_string is None or best_chromosome.binary_string is None:
            raise ValueError("Chromosome has not been measured (measure method not called)!")
            
        x = chromosome.binary_string
        b = best_chromosome.binary_string
        
        delta_theta = np.zeros(chromosome.n_genes, dtype=np.float64)
        
        # Algorithmic logic: If current bit implies '0' and best is '1', rotate positively
        #                    If current bit implies '1' and best is '0', rotate negatively
        #                    Otherwise, leave the angle intact (maintain superposition)
        mask_01 = (x == 0) & (b == 1)
        mask_10 = (x == 1) & (b == 0)
        
        delta_theta[mask_01] = theta
        delta_theta[mask_10] = -theta
        
        # Vectorized Multiplication of the 2x2 Unitary Rotation Matrix
        # [ alpha' ] = [ cos(th) -sin(th) ] [ alpha ]
        # [ beta'  ]   [ sin(th)  cos(th) ] [ beta  ]
        cos_t = np.cos(delta_theta)
        sin_t = np.sin(delta_theta)
        
        new_alpha = cos_t * chromosome.alpha - sin_t * chromosome.beta
        new_beta  = sin_t * chromosome.alpha + cos_t * chromosome.beta
        
        chromosome.alpha = new_alpha
        chromosome.beta = new_beta
        chromosome._normalize()

class DQGA:
    """
    Discrete Quantum Genetic Optimizer.
    
    Attributes:
        pop_size (int): Number of individuals in the population.
        n_genes (int): Number of qubits per chromosome.
        fitness_fn (Callable): A generic function accepting a 1D binary array (np.ndarray)
                               and returning a fitness score (float).
        n_generations (int): Evolutionary loop cycle limit.
    """
    
    def __init__(
        self,
        pop_size: int,
        n_genes: int,
        fitness_fn: Callable[[np.ndarray], float],
        n_generations: int = 50,
        theta: float = 0.05 * np.pi
    ) -> None:
        assert pop_size > 0
        assert n_genes > 0
        
        self.pop_size = pop_size
        self.n_genes = n_genes
        self.fitness_fn = fitness_fn
        self.n_generations = n_generations
        self.theta = theta
        
        self.population: List[QuantumChromosome] = [QuantumChromosome(n_genes) for _ in range(pop_size)]
        self.global_best_chromosome: Optional[QuantumChromosome] = None
        
        # History logs
        self.history_best_fitness: List[float] = []

    def _evaluate_population(self) -> None:
        """Measures the entire population and routes it through the fitness function."""
        for chrom in self.population:
            # 1. Fetch the classical bit observation (Measurement)
            bits = chrom.measure()
            
            # 2. Score fitness
            chrom.fitness = self.fitness_fn(bits)
            
            # Update Global Best
            if self.global_best_chromosome is None or chrom.fitness > self.global_best_chromosome.fitness:
                # Retain via explicit copy to avoid reference overwrites down the line
                best_copy = QuantumChromosome(self.n_genes)
                best_copy.alpha = chrom.alpha.copy()
                best_copy.beta = chrom.beta.copy()
                best_copy.binary_string = chrom.binary_string.copy()
                best_copy.fitness = chrom.fitness
                self.global_best_chromosome = best_copy

    def run(self) -> Tuple[np.ndarray, float]:
        """
        Initiates the Evolutionary algorithm.
        
        Returns:
            Tuple[np.ndarray, float]: The optimal classical bit sequence and its maximum fitness score.
        """
        logger.info("Initializing DQGA: pop=%d, genes=%d, gen=%d", self.pop_size, self.n_genes, self.n_generations)
        
        self.history_best_fitness = []
        
        for gen in range(self.n_generations):
            # 1. Measurement and Evaluation
            self._evaluate_population()
            
            best_fit = self.global_best_chromosome.fitness
            self.history_best_fitness.append(best_fit)
            
            # 2. Quantum Rotation (Evolutionary Step)
            for chrom in self.population:
                QuantumGate.rotate(chrom, self.global_best_chromosome, self.theta)
                
            if (gen + 1) % 10 == 0:
                logger.debug("DQGA Generation %d | Best Fitness: %.4f", gen+1, best_fit)
                
        logger.info("DQGA Complete. Max Fitness: %.4f", self.global_best_chromosome.fitness)
        return self.global_best_chromosome.binary_string, self.global_best_chromosome.fitness

# Value Mapping Intermediary (For routing DQGA bits into float bounds)
def bits_to_float(bits: np.ndarray, val_min: float, val_max: float) -> float:
    """
    Linearly maps a binary bit string representation into the [val_min, val_max] float parameter bound.
    """
    n_bits = len(bits)
    # Binary to Decimal transposition (Vectorized)
    powers = 2 ** np.arange(n_bits - 1, -1, -1)
    decimal_val = np.sum(bits * powers)
    
    max_decimal = (2 ** n_bits) - 1
    if max_decimal <= 0:
        return val_min
        
    normalized = decimal_val / max_decimal
    return val_min + normalized * (val_max - val_min)
