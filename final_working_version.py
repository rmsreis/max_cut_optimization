#!/usr/bin/env python3
"""
Working IBM Quantum Max Cut - Final Version
===========================================

This version fixes all API issues and works with current Qiskit Runtime.
Perfect for educational demonstrations with real IBM Quantum devices.
"""

import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit
from qiskit_ibm_runtime import QiskitRuntimeService, Sampler
from qiskit.circuit import Parameter
import rustworkx as rx
from itertools import combinations
import time

def save_ibm_token():
    """Get and save IBM Quantum token from user."""
    print("🔑 IBM QUANTUM TOKEN SETUP")
    print("=" * 30)
    
    try:
        service = QiskitRuntimeService(channel="ibm_cloud")
        print("✅ Token already saved and working!")
        return True
    except:
        print("Need to save your IBM Quantum token first.")
        token = input("Enter your IBM Quantum token: ").strip()
        
        try:
            QiskitRuntimeService.save_account(
                channel="ibm_cloud",
                token=token,
                overwrite=True
            )
            print("✅ Token saved successfully!")
            return True
        except Exception as e:
            print(f"❌ Error saving token: {e}")
            return False

def create_sample_graph(n_nodes=5):
    """Create a sample graph for max cut problem."""
    G = rx.PyGraph()
    
    # Add nodes
    node_indices = []
    for i in range(n_nodes):
        node_indices.append(G.add_node(i))
    
    # Create edges based on number of nodes
    edges = []
    
    if n_nodes == 3:
        # Simple triangle
        edges = [(0, 1), (1, 2), (2, 0)]
    elif n_nodes == 4:
        # Square with diagonal
        edges = [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)]
    elif n_nodes == 5:
        # Pentagon with one internal edge
        edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0), (1, 3)]
    elif n_nodes == 6:
        # Hexagon with some internal connections
        edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0), (0, 3), (1, 4)]
    else:
        # For other sizes, create a cycle + some random connections
        # Create cycle
        for i in range(n_nodes):
            edges.append((i, (i + 1) % n_nodes))
        
        # Add some additional edges for more interesting cuts
        import random
        random.seed(42)  # For reproducible results
        additional_edges = min(n_nodes // 2, 3)  # Add some extra edges
        for _ in range(additional_edges):
            i, j = random.sample(range(n_nodes), 2)
            if (i, j) not in edges and (j, i) not in edges and i != j:
                edges.append((i, j))
    
    # Add edges to graph
    for i, j in edges:
        G.add_edge(i, j, 1.0)  # weight = 1.0
    
    return G, edges

def classical_max_cut_brute_force(graph, edges):
    """Find maximum cut using brute force (for small graphs)."""
    n = len(graph.node_indices())
    max_cut_value = 0
    best_partition = None
    
    # Try all possible partitions (2^n possibilities)
    for i in range(1, 2**n - 1):  # Exclude empty and full partitions
        partition_s = []
        partition_t = []
        
        for node in range(n):
            if i & (1 << node):
                partition_s.append(node)
            else:
                partition_t.append(node)
        
        # Calculate cut value
        cut_value = 0
        for edge in edges:
            u, v = edge
            if (u in partition_s and v in partition_t) or (u in partition_t and v in partition_s):
                cut_value += 1
        
        if cut_value > max_cut_value:
            max_cut_value = cut_value
            best_partition = (partition_s, partition_t)
    
    return max_cut_value, best_partition

def create_qaoa_circuit(graph, edges, gamma, beta):
    """Create QAOA circuit for max cut problem."""
    n = len(graph.node_indices())
    
    # Create quantum circuit
    qc = QuantumCircuit(n, n)
    
    # Initial state: |+⟩^n (superposition)
    qc.h(range(n))
    
    # Cost layer (problem Hamiltonian)
    for edge in edges:
        i, j = edge
        qc.cx(i, j)
        qc.rz(2 * gamma, j)
        qc.cx(i, j)
    
    # Mixer layer (mixing Hamiltonian)
    for i in range(n):
        qc.rx(2 * beta, i)
    
    # Measurement
    qc.measure_all()
    
    return qc

def run_qaoa_modern(graph, edges, device_name=None):
    """
    Run QAOA using modern Qiskit Runtime primitives.
    This replaces the deprecated backend.run() method.
    """
    print(f"\n🚀 RUNNING QAOA WITH MODERN RUNTIME PRIMITIVES")
    print("=" * 50)
    
    try:
        # Connect to IBM Quantum
        service = QiskitRuntimeService(channel="ibm_cloud")
        
        # Select device
        n_qubits_needed = len(graph.node_indices())
        
        if device_name:
            backend = service.backend(device_name)
            print(f"🎯 Using specified device: {device_name}")
        else:
            # Use least busy device with enough qubits
            backend = service.least_busy(operational=True, min_num_qubits=n_qubits_needed)
            print(f"🎯 Using least busy device: {backend.name}")
        
        # Check device status
        status = backend.status()
        config = backend.configuration()
        print(f"   • Qubits: {config.n_qubits}")
        print(f"   • Queue: {status.pending_jobs} jobs")
        
        # QAOA parameters (simple optimization)
        gamma = np.pi / 4  # Cost parameter
        beta = np.pi / 2   # Mixer parameter
        
        # Create QAOA circuit
        qc = create_qaoa_circuit(graph, edges, gamma, beta)
        print(f"   • Circuit depth: {qc.depth()}")
        print(f"   • Circuit gates: {qc.count_ops()}")
        
        # Run using Sampler (modern approach) - No Session needed for simple jobs
        sampler = Sampler(backend=backend)
        
        print(f"\n⏳ Submitting job to {backend.name}...")
        print(f"   Expected wait time: ~{status.pending_jobs * 2} minutes")
        
        # Submit job
        job = sampler.run([qc], shots=1024)
        
        print(f"✅ Job submitted! Job ID: {job.job_id()}")
        print(f"🔄 Job status: {job.status()}")
        
        # Wait for completion
        print("⏳ Waiting for job to complete...")
        start_time = time.time()
        
        while job.status() not in ['DONE', 'FAILED', 'CANCELLED']:
            elapsed = int(time.time() - start_time)
            print(f"   ⏱️  Elapsed: {elapsed}s, Status: {job.status()}")
            time.sleep(10)
        
        if job.status() == 'DONE':
            print("✅ Job completed successfully!")
            
            # Get results
            result = job.result()
            counts = result[0].data.meas.get_counts()
            
            # Analyze results
            print(f"\n📊 QUANTUM RESULTS (top 5 solutions):")
            sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
            
            best_cut_value = 0
            best_bitstring = None
            
            for i, (bitstring, count) in enumerate(sorted_counts[:5]):
                # Calculate cut value for this bitstring
                cut_value = calculate_cut_value(bitstring, edges)
                probability = count / 1024 * 100
                
                if cut_value > best_cut_value:
                    best_cut_value = cut_value
                    best_bitstring = bitstring
                
                print(f"   {i+1}. {bitstring} → Cut: {cut_value}, Prob: {probability:.1f}%")
            
            return best_cut_value, best_bitstring, counts
            
        else:
            print(f"❌ Job failed with status: {job.status()}")
            return None, None, None
            
    except Exception as e:
        print(f"❌ Error running QAOA: {e}")
        print(f"🔧 Trying local simulation instead...")
        return run_qaoa_simulation(graph, edges)

def calculate_cut_value(bitstring, edges):
    """Calculate cut value for a given bitstring."""
    cut_value = 0
    for edge in edges:
        i, j = edge
        # If nodes are in different partitions (different bit values)
        if bitstring[i] != bitstring[j]:
            cut_value += 1
    return cut_value

def run_qaoa_simulation(graph, edges):
    """Run QAOA on local simulator as fallback."""
    from qiskit_aer import AerSimulator
    
    print("🖥️  RUNNING ON LOCAL SIMULATOR")
    print("-" * 35)
    
    simulator = AerSimulator()
    gamma = np.pi / 4
    beta = np.pi / 2
    
    qc = create_qaoa_circuit(graph, edges, gamma, beta)
    
    # Run simulation
    job = simulator.run(qc, shots=1024)
    result = job.result()
    counts = result.get_counts()
    
    # Find best solution
    best_cut_value = 0
    best_bitstring = None
    
    print("📊 Top 5 simulation results:")
    sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    
    for i, (bitstring, count) in enumerate(sorted_counts[:5]):
        cut_value = calculate_cut_value(bitstring, edges)
        probability = count / 1024 * 100
        
        if cut_value > best_cut_value:
            best_cut_value = cut_value
            best_bitstring = bitstring
        
        print(f"   {i+1}. {bitstring} → Cut: {cut_value}, Prob: {probability:.1f}%")
    
    return best_cut_value, best_bitstring, counts

def visualize_graph_and_cut(graph, edges, classical_partition, quantum_bitstring=None):
    """Visualize the graph with classical and quantum solutions."""
    fig, axes = plt.subplots(1, 2 if quantum_bitstring else 1, figsize=(12, 5))
    if quantum_bitstring is None:
        axes = [axes]
    
    # Generate node positions based on graph size
    n_nodes = len(graph.node_indices())
    pos = generate_node_positions(n_nodes)
    
    # Classical solution
    ax1 = axes[0]
    
    # Draw nodes
    partition_s, partition_t = classical_partition
    for node in partition_s:
        ax1.scatter(*pos[node], c='red', s=200, label='Set S' if node == partition_s[0] else "")
    for node in partition_t:
        ax1.scatter(*pos[node], c='blue', s=200, label='Set T' if node == partition_t[0] else "")
    
    # Draw edges
    for edge in edges:
        i, j = edge
        x_coords = [pos[i][0], pos[j][0]]
        y_coords = [pos[i][1], pos[j][1]]
        
        # Red if edge crosses the cut, gray otherwise
        if (i in partition_s and j in partition_t) or (i in partition_t and j in partition_s):
            ax1.plot(x_coords, y_coords, 'r-', linewidth=3, alpha=0.7)
        else:
            ax1.plot(x_coords, y_coords, 'gray', linewidth=1, alpha=0.5)
    
    # Add node labels
    for node, position in pos.items():
        ax1.text(position[0], position[1], str(node), ha='center', va='center', 
                fontsize=12, fontweight='bold', color='white')
    
    ax1.set_title('Classical Optimal Solution', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    
    # Quantum solution (if available)
    if quantum_bitstring:
        ax2 = axes[1]
        
        # Parse quantum bitstring (fix indexing issue)
        n_qubits = len(quantum_bitstring)
        quantum_s = [i for i in range(n_qubits) if quantum_bitstring[i] == '1']
        quantum_t = [i for i in range(n_qubits) if quantum_bitstring[i] == '0']
        
        # Draw nodes
        for node in quantum_s:
            if node in pos:  # Safety check
                ax2.scatter(*pos[node], c='orange', s=200, label='Set S' if len([x for x in quantum_s if x == node]) == 1 else "")
        for node in quantum_t:
            if node in pos:  # Safety check
                ax2.scatter(*pos[node], c='green', s=200, label='Set T' if len([x for x in quantum_t if x == node]) == 1 else "")
        
        # Draw edges
        for edge in edges:
            i, j = edge
            x_coords = [pos[i][0], pos[j][0]]
            y_coords = [pos[i][1], pos[j][1]]
            
            # Orange if edge crosses the cut
            if (i in quantum_s and j in quantum_t) or (i in quantum_t and j in quantum_s):
                ax2.plot(x_coords, y_coords, 'orange', linewidth=3, alpha=0.7)
            else:
                ax2.plot(x_coords, y_coords, 'gray', linewidth=1, alpha=0.5)
        
        # Add node labels
        for node, position in pos.items():
            ax2.text(position[0], position[1], str(node), ha='center', va='center', 
                    fontsize=12, fontweight='bold', color='white')
        
        ax2.set_title('Quantum QAOA Solution', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_aspect('equal')
    
    plt.tight_layout()
    plt.show()

def generate_node_positions(n_nodes):
    """Generate node positions for visualization based on number of nodes."""
    import math
    
    pos = {}
    
    if n_nodes <= 3:
        # Triangle layout
        angles = [2 * math.pi * i / n_nodes for i in range(n_nodes)]
        for i in range(n_nodes):
            pos[i] = (math.cos(angles[i]), math.sin(angles[i]))
    
    elif n_nodes == 4:
        # Square layout
        pos = {0: (-1, 1), 1: (1, 1), 2: (1, -1), 3: (-1, -1)}
    
    elif n_nodes == 5:
        # Pentagon layout (original)
        pos = {0: (0, 1), 1: (1, 1), 2: (2, 0), 3: (1, -1), 4: (0, 0)}
    
    elif n_nodes == 6:
        # Hexagon layout
        angles = [2 * math.pi * i / 6 for i in range(6)]
        for i in range(6):
            pos[i] = (1.5 * math.cos(angles[i]), 1.5 * math.sin(angles[i]))
    
    else:
        # Circular layout for larger graphs
        radius = max(1.5, n_nodes / 4)  # Scale radius with number of nodes
        angles = [2 * math.pi * i / n_nodes for i in range(n_nodes)]
        for i in range(n_nodes):
            pos[i] = (radius * math.cos(angles[i]), radius * math.sin(angles[i]))
    
    return pos

def main():
    """Main demonstration function."""
    print("🌟 WORKING IBM QUANTUM MAX CUT DEMONSTRATION")
    print("=" * 48)
    
    # Setup token
    if not save_ibm_token():
        return
    
    # Select problem size
    print(f"\n📊 PROBLEM SIZE SELECTION")
    print("-" * 26)
    print("Choose number of nodes:")
    print("1. 🔹 3 nodes (simple triangle)")
    print("2. 🔸 4 nodes (square with diagonal)")
    print("3. 🔹 5 nodes (pentagon + internal edge)")
    print("4. 🔸 6 nodes (hexagon with connections)")
    print("5. 🔹 Custom size (3-12 nodes)")
    
    size_choice = input("Choice (1-5): ").strip()
    
    if size_choice == "1":
        n_nodes = 3
    elif size_choice == "2":
        n_nodes = 4
    elif size_choice == "3":
        n_nodes = 5
    elif size_choice == "4":
        n_nodes = 6
    elif size_choice == "5":
        while True:
            try:
                n_nodes = int(input("Enter number of nodes (3-12): "))
                if 3 <= n_nodes <= 12:
                    break
                else:
                    print("Please enter a number between 3 and 12")
            except ValueError:
                print("Please enter a valid number")
    else:
        n_nodes = 5  # Default
        print("Using default: 5 nodes")
    
    # Create sample problem
    print(f"\n📊 CREATING {n_nodes}-NODE GRAPH")
    print("-" * (19 + len(str(n_nodes))))
    graph, edges = create_sample_graph(n_nodes)
    print(f"✅ Created {n_nodes}-node graph with {len(edges)} edges")
    print(f"   Edges: {edges}")
    
    # Warning for larger problems
    if n_nodes > 8:
        print(f"⚠️  Note: {n_nodes} nodes = {2**n_nodes} possible partitions")
        print("   Classical brute force may take longer...")
    
    # Classical solution
    print(f"\n🧮 CLASSICAL SOLUTION")
    print("-" * 20)
    
    if n_nodes <= 10:  # Reasonable limit for brute force
        classical_max_cut, classical_partition = classical_max_cut_brute_force(graph, edges)
        print(f"✅ Classical optimal cut value: {classical_max_cut}")
        print(f"   Partition S: {classical_partition[0]}")
        print(f"   Partition T: {classical_partition[1]}")
    else:
        print(f"⚠️  Skipping brute force for {n_nodes} nodes (too many combinations)")
        print("   Using greedy approximation instead...")
        classical_max_cut, classical_partition = greedy_max_cut(graph, edges)
        print(f"✅ Greedy approximation cut value: {classical_max_cut}")
        print(f"   Partition S: {classical_partition[0]}")
        print(f"   Partition T: {classical_partition[1]}")
    
    # Device selection
    print(f"\n🎯 DEVICE SELECTION")
    print("-" * 20)
    print(f"Required qubits: {n_nodes}")
    print("Available options:")
    print("1. 🤖 Auto (least busy)")
    print("2. 🎯 ibm_fez (156 qubits, ~69 jobs)")
    print("3. 🎯 ibm_kingston (156 qubits, ~69 jobs)")
    print("4. 🖥️  Local simulator (fast)")
    
    if n_nodes > 127:
        print(f"⚠️  Warning: {n_nodes} qubits may exceed some device limits")
    
    device_choice = input("Choice (1-4): ").strip()
    
    device_name = None
    if device_choice == "2":
        device_name = "ibm_fez"
    elif device_choice == "3":
        device_name = "ibm_kingston"
    elif device_choice == "4":
        quantum_cut, quantum_bitstring, _ = run_qaoa_simulation(graph, edges)
    else:
        device_name = None  # Auto-select
    
    # Run quantum solution (if not simulator)
    if device_choice != "4":
        quantum_cut, quantum_bitstring, counts = run_qaoa_modern(graph, edges, device_name)
    
    # Results comparison
    print(f"\n📈 RESULTS COMPARISON")
    print("=" * 25)
    print(f"Classical optimal: {classical_max_cut}")
    if quantum_cut:
        print(f"Quantum QAOA:      {quantum_cut}")
        approximation_ratio = quantum_cut / classical_max_cut
        print(f"Approximation ratio: {approximation_ratio:.2%}")
        
        if approximation_ratio >= 0.8:
            print("🏆 Excellent quantum performance!")
        elif approximation_ratio >= 0.6:
            print("✅ Good quantum approximation")
        else:
            print("⚠️  Room for improvement")
    
    # Visualization
    print(f"\n📊 CREATING VISUALIZATION")
    visualize_graph_and_cut(graph, edges, classical_partition, quantum_bitstring)
    
    print(f"\n🎉 Demonstration complete!")
    print(f"📚 This shows both classical and quantum approaches to Max Cut")
    print(f"🚀 Perfect for teaching quantum computing concepts!")
    print(f"📏 Problem size: {n_nodes} nodes, {len(edges)} edges")

def greedy_max_cut(graph, edges):
    """Greedy approximation for max cut (for larger graphs)."""
    n = len(graph.node_indices())
    partition_s = []
    partition_t = list(range(n))
    
    improved = True
    while improved:
        improved = False
        best_gain = 0
        best_move = None
        
        # Try moving each node to the other partition
        for node in range(n):
            current_set = 's' if node in partition_s else 't'
            
            # Calculate gain from moving this node
            gain = 0
            for edge in edges:
                i, j = edge
                if i == node or j == node:
                    other_node = j if i == node else i
                    
                    if current_set == 's':
                        # Moving from S to T
                        if other_node in partition_s:
                            gain -= 1  # Lose this cut
                        else:
                            gain += 1  # Gain this cut
                    else:
                        # Moving from T to S
                        if other_node in partition_t:
                            gain -= 1  # Lose this cut
                        else:
                            gain += 1  # Gain this cut
            
            if gain > best_gain:
                best_gain = gain
                best_move = (node, current_set)
        
        # Make the best move
        if best_move and best_gain > 0:
            node, current_set = best_move
            if current_set == 's':
                partition_s.remove(node)
                partition_t.append(node)
            else:
                partition_t.remove(node)
                partition_s.append(node)
            improved = True
    
    # Calculate final cut value
    cut_value = 0
    for edge in edges:
        i, j = edge
        if (i in partition_s and j in partition_t) or (i in partition_t and j in partition_s):
            cut_value += 1
    
    return cut_value, (partition_s, partition_t)

if __name__ == "__main__":
    main()
