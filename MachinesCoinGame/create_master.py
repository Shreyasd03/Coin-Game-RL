#!/usr/bin/env python3
"""
Quick script to create a master model from current training.
Usage: python create_master.py [master_name]
"""

import sys
import os
import shutil
from datetime import datetime

def create_master(master_name=None):
    """Create a master model from current training."""
    
    # Check if current model exists
    current_model = "models/ppo_coin_final.zip"
    current_vecnormalize = "models/vecnormalize.pkl"
    
    if not os.path.exists(current_model):
        print("❌ No current model found. Train a model first.")
        return False
    
    # Get master name
    if not master_name:
        master_name = input("Enter master model name (e.g., 'ground_master', 'jump_master'): ").strip()
        if not master_name:
            print("❌ Master name cannot be empty")
            return False
    
    # Create masterModels directory if it doesn't exist
    os.makedirs("masterModels", exist_ok=True)
    
    # Save master model
    master_model_path = f"masterModels/ppo_coin_{master_name}.zip"
    master_vecnormalize_path = f"masterModels/vecnormalize_{master_name}.pkl"
    
    # Copy model
    shutil.copy(current_model, master_model_path)
    print(f"✅ Master model created: {master_model_path}")
    
    # Copy VecNormalize if it exists
    if os.path.exists(current_vecnormalize):
        shutil.copy(current_vecnormalize, master_vecnormalize_path)
        print(f"✅ Master VecNormalize created: {master_vecnormalize_path}")
    else:
        print("⚠️  VecNormalize not found - you may need to retrain for proper normalization")
    
    print(f"\n🎯 Master model '{master_name}' is ready!")
    print("   You can now use this master model for future training.")
    
    return True

if __name__ == "__main__":
    master_name = sys.argv[1] if len(sys.argv) > 1 else None
    create_master(master_name)
