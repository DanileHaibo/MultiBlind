#!/usr/bin/env python3
"""
Script to read and display attack results in formatted tables.
Reads map and planning results from JSON files and displays them in tables
similar to the paper format.

Usage:
    python print_attack_results.py

The script will automatically find attack result directories in dataset/maptr-bevpool/
and display formatted tables for:
1. RSA attack map results (AP percentages)
2. RSA attack planning results (unreachable goal rates) 
3. ETA attack map results (AP percentages)
4. ETA attack planning results (unreachable goal rates)

Expected directory structure:
    dataset/maptr-bevpool/
    ├── train_blind_rsa_asymmetric/     # RSA blinding attack results
    ├── train_patch_rsa_asymmetric/     # RSA patch attack results  
    ├── train_blind_eta_asymmetric/     # ETA blinding attack results
    └── train_patch_eta_asymmetric/     # ETA patch attack results

Each attack directory should contain:
    results/map/clean/mAPs.json         # Clean baseline map results
    results/map/attack/mAPs.json        # Attack map results
    results/planning/summary.json       # Planning results with unreachable goal rates

Example output:
    🚗 Attack Results Summary
    ✅ Found results for: rsa_blind, eta_blind
    
    Table: Map AP(%) on asymmetric scenes under RSA attacks.
    
    Blinding (Black-box) Attack:
    +------------+-------------+------------+--------+------+
    | Method     | AP_boundary | AP_divider | AP_ped | mAP  |
    +------------+-------------+------------+--------+------+
    | Clean      |     48.9    |    54.2    |  38.2  | 47.1 |
    | RSA (Ours) |     40.2    |    44.1    |  36.1  | 40.1 |
    +------------+-------------+------------+--------+------+
    
    Table: Unreachable Goal Rate (%) on asymmetric scenes under RSA attacks.
    Table: Unsafe Planned Trajectory Rate (%) on asymmetric scenes under ETA attacks.
"""

import json
import os
from pathlib import Path
from typing import Dict, Optional, Tuple
from prettytable import PrettyTable


class AttackResultsReader:
    def __init__(self, base_path: str = "dataset/maptr-bevpool"):
        self.base_path = Path(base_path)
        
        # Define attack directory mappings
        self.attack_dirs = {
            'rsa_blind': 'train_blind_rsa_asymmetric',
            'rsa_patch': 'train_patch_rsa_asymmetric', 
            'eta_blind': 'train_blind_eta_asymmetric',
            'eta_patch': 'train_patch_eta_asymmetric'
        }
    
    def load_map_results(self, attack_dir: str) -> Tuple[Optional[Dict], Optional[Dict]]:
        """Load clean and attack map results from mAPs.json files."""
        dir_path = self.base_path / attack_dir
        
        clean_path = dir_path / "results" / "map" / "clean" / "mAPs.json"
        attack_path = dir_path / "results" / "map" / "attack" / "mAPs.json"
        
        clean_results = None
        attack_results = None
        
        try:
            if clean_path.exists():
                with open(clean_path, 'r') as f:
                    clean_results = json.load(f)
        except Exception as e:
            print(f"Warning: Could not load clean results from {clean_path}: {e}")
        
        try:
            if attack_path.exists():
                with open(attack_path, 'r') as f:
                    attack_results = json.load(f)
        except Exception as e:
            print(f"Warning: Could not load attack results from {attack_path}: {e}")
        
        return clean_results, attack_results
    
    def load_planning_results(self, attack_dir: str) -> Optional[Dict]:
        """Load planning results from summary.json file."""
        dir_path = self.base_path / attack_dir
        planning_path = dir_path / "results" / "planning" / "summary.json"
        
        try:
            if planning_path.exists():
                with open(planning_path, 'r') as f:
                    return json.load(f)
        except Exception as e:
            print(f"Warning: Could not load planning results from {planning_path}: {e}")
        
        return None
    
    def extract_map_metrics(self, results: Dict) -> Dict[str, float]:
        """Extract relevant map metrics and convert to percentages."""
        if not results:
            return {"boundary": 0.0, "divider": 0.0, "ped": 0.0, "mAP": 0.0}
        
        return {
            "boundary": results.get("NuscMap_chamfer/boundary_AP", 0.0) * 100,
            "divider": results.get("NuscMap_chamfer/divider_AP", 0.0) * 100, 
            "ped": results.get("NuscMap_chamfer/ped_crossing_AP", 0.0) * 100,
            "mAP": results.get("NuscMap_chamfer/mAP", 0.0) * 100
        }
    
    def extract_planning_metric(self, results: Dict, attack_type: str = "rsa") -> Tuple[float, float]:
        """Extract planning metrics for clean and attack."""
        if not results:
            return 0.0, 0.0
        
        # For RSA attacks, focus on unreachable goal rate
        # For ETA attacks, focus on unsafe planned trajectory rate
        if attack_type.lower() == "eta":
            metric_key = "unsafe_planned_trajectory_rate"
        else:
            metric_key = "unreachable_goal_rate"
            
        metric_data = results.get(metric_key, {})
        clean_rate = metric_data.get("clean", 0.0) * 100
        attack_rate = metric_data.get("attack", 0.0) * 100
        
        return clean_rate, attack_rate
    
    def print_map_table(self, attack_type: str, rsa_or_eta: str):
        """Print formatted map AP table for given attack type."""
        print(f"\nTable: Map AP(%) on asymmetric scenes under {rsa_or_eta.upper()} attacks.")
        
        # Get results for blinding and patch attacks
        blind_dir = self.attack_dirs[f'{rsa_or_eta}_blind']
        patch_dir = self.attack_dirs[f'{rsa_or_eta}_patch']
        
        # Check if directories exist
        blind_exists = (self.base_path / blind_dir).exists()
        patch_exists = (self.base_path / patch_dir).exists()
        
        if not blind_exists and not patch_exists:
            print(f"No {rsa_or_eta.upper()} attack results found.")
            print(f"Expected directories: {blind_dir}, {patch_dir}")
            return
        
        # Load results
        blind_clean, blind_attack = self.load_map_results(blind_dir) if blind_exists else (None, None)
        patch_clean, patch_attack = self.load_map_results(patch_dir) if patch_exists else (None, None)
        
        # Extract metrics  
        blind_clean_metrics = self.extract_map_metrics(blind_clean)
        blind_attack_metrics = self.extract_map_metrics(blind_attack)
        patch_clean_metrics = self.extract_map_metrics(patch_clean)
        patch_attack_metrics = self.extract_map_metrics(patch_attack)
        
        # Create separate tables for blinding and patch attacks
        if blind_exists:
            print("\nBlinding (Black-box) Attack:")
            blind_table = PrettyTable()
            blind_table.field_names = ["Method", "AP_boundary", "AP_divider", "AP_ped", "mAP"]
            blind_table.align = "c"
            blind_table.align["Method"] = "l"
            
            # Add Clean row
            blind_table.add_row([
                "Clean",
                f"{blind_clean_metrics['boundary']:.1f}",
                f"{blind_clean_metrics['divider']:.1f}",
                f"{blind_clean_metrics['ped']:.1f}",
                f"{blind_clean_metrics['mAP']:.1f}"
            ])
            
            # Add Attack row
            blind_table.add_row([
                f"{rsa_or_eta.upper()} (Ours)",
                f"{blind_attack_metrics['boundary']:.1f}",
                f"{blind_attack_metrics['divider']:.1f}",
                f"{blind_attack_metrics['ped']:.1f}",
                f"{blind_attack_metrics['mAP']:.1f}"
            ])
            
            print(blind_table)
        
        if patch_exists:
            print("\nAdversarial Patch (White-box) Attack:")
            patch_table = PrettyTable()
            patch_table.field_names = ["Method", "AP_boundary", "AP_divider", "AP_ped", "mAP"]
            patch_table.align = "c"
            patch_table.align["Method"] = "l"
            
            # Add Clean row
            patch_table.add_row([
                "Clean",
                f"{patch_clean_metrics['boundary']:.1f}",
                f"{patch_clean_metrics['divider']:.1f}",
                f"{patch_clean_metrics['ped']:.1f}",
                f"{patch_clean_metrics['mAP']:.1f}"
            ])
            
            # Add Attack row
            patch_table.add_row([
                f"{rsa_or_eta.upper()} (Ours)",
                f"{patch_attack_metrics['boundary']:.1f}",
                f"{patch_attack_metrics['divider']:.1f}",
                f"{patch_attack_metrics['ped']:.1f}",
                f"{patch_attack_metrics['mAP']:.1f}"
            ])
            
            print(patch_table)
    
    def print_planning_table(self, attack_type: str, rsa_or_eta: str):
        """Print formatted planning results table for given attack type."""
        # Different metrics for different attack types
        if rsa_or_eta.lower() == "eta":
            metric_name = "Unsafe Planned Trajectory Rate"
        else:
            metric_name = "Unreachable Goal Rate"
            
        print(f"\nTable: {metric_name} (%) on asymmetric scenes under {rsa_or_eta.upper()} attacks.")
        
        # Get results for blinding and patch attacks
        blind_dir = self.attack_dirs[f'{rsa_or_eta}_blind']
        patch_dir = self.attack_dirs[f'{rsa_or_eta}_patch']
        
        # Check if directories exist
        blind_exists = (self.base_path / blind_dir).exists()
        patch_exists = (self.base_path / patch_dir).exists()
        
        if not blind_exists and not patch_exists:
            print(f"No {rsa_or_eta.upper()} attack results found.")
            print(f"Expected directories: {blind_dir}, {patch_dir}")
            return
        
        # Load planning results
        blind_planning = self.load_planning_results(blind_dir) if blind_exists else None
        patch_planning = self.load_planning_results(patch_dir) if patch_exists else None
        
        # Extract planning metrics
        blind_clean_rate, blind_attack_rate = self.extract_planning_metric(blind_planning, rsa_or_eta) if blind_exists else (0, 0)
        patch_clean_rate, patch_attack_rate = self.extract_planning_metric(patch_planning, rsa_or_eta) if patch_exists else (0, 0)
        
        # Create PrettyTable
        table = PrettyTable()
        table.field_names = ["Method", "Blinding (%)", "Patch (%)"]
        
        # Set alignment
        table.align = "c"
        table.align["Method"] = "l"
        
        # Add Clean row
        clean_row = [
            "Clean",
            f"{blind_clean_rate:.0f}" if blind_exists else "N/A",
            f"{patch_clean_rate:.0f}" if patch_exists else "N/A"
        ]
        table.add_row(clean_row)
        
        # Add Attack row with increase
        blind_attack_str = "N/A"
        patch_attack_str = "N/A"
        
        if blind_exists:
            blind_increase = blind_attack_rate - blind_clean_rate
            blind_attack_str = f"{blind_attack_rate:.0f} (+{blind_increase:.0f})"
            
        if patch_exists:
            patch_increase = patch_attack_rate - patch_clean_rate
            patch_attack_str = f"{patch_attack_rate:.0f} (+{patch_increase:.0f})"
        
        attack_row = [
            f"{rsa_or_eta.upper()} (Ours)",
            blind_attack_str,
            patch_attack_str
        ]
        table.add_row(attack_row)
        
        print(table)
    
    def print_all_results(self):
        """Print all 4 tables: RSA map, RSA planning, ETA map, ETA planning."""
        print("🚗 Attack Results Summary")
        print("=" * 80)
        
        # Check which attack directories exist
        existing_attacks = []
        for key, dir_name in self.attack_dirs.items():
            dir_path = self.base_path / dir_name
            if dir_path.exists():
                existing_attacks.append(key)
        
        if not existing_attacks:
            print("❌ No attack result directories found!")
            print(f"Expected directories in {self.base_path}:")
            for key, dir_name in self.attack_dirs.items():
                print(f"  - {dir_name}")
            return
        
        print(f"✅ Found results for: {', '.join(existing_attacks)}")
        
        # RSA Results
        print("\n" + "🎯 RSA ATTACK RESULTS".center(80, "="))
        
        # RSA Map Results
        self.print_map_table("rsa", "rsa")
        
        # RSA Planning Results  
        self.print_planning_table("rsa", "rsa")
        
        # ETA Results
        print("\n" + "🎯 ETA ATTACK RESULTS".center(80, "="))
        
        # ETA Map Results
        self.print_map_table("eta", "eta")
        
        # ETA Planning Results
        self.print_planning_table("eta", "eta")


def main():
    """Main function to run the results display."""
    # Create reader instance
    reader = AttackResultsReader()
    
    # Print all results
    reader.print_all_results()


if __name__ == "__main__":
    main()
