#!/usr/bin/env python3
"""
Restore ground model for evaluation or continued training.
This script copies the saved ground model back to the main model files.
"""

import shutil
import os

def restore_ground_model():
    """Restore the ground model to main model files."""
    
    ground_model = "masterModels/ppo_coin_ground_master.zip"
    ground_vecnormalize = "masterModels/vecnormalize_ground.pkl"
    
    main_model = "models/ppo_coin_final.zip"
    main_vecnormalize = "models/vecnormalize.pkl"
    
    if not os.path.exists(ground_model):
        print("❌ Ground model not found! Run training first to create ground model.")
        return False
        
    if not os.path.exists(ground_vecnormalize):
        print("❌ Ground VecNormalize not found! Run training first to create ground model.")
        return False
    
    # Backup current model if it exists
    if os.path.exists(main_model):
        shutil.copy(main_model, "models/ppo_coin_backup.zip")
        print("📦 Backed up current model to ppo_coin_backup.zip")
    
    if os.path.exists(main_vecnormalize):
        shutil.copy(main_vecnormalize, "models/vecnormalize_backup.pkl")
        print("📦 Backed up current VecNormalize to vecnormalize_backup.pkl")
    
    # Restore ground model
    shutil.copy(ground_model, main_model)
    shutil.copy(ground_vecnormalize, main_vecnormalize)
    
    print("✅ Ground model restored!")
    print("🎯 You can now:")
    print("   - Run visual evaluation: python -m rl.visualEval")
    print("   - Continue training: python -m rl.trainPPO")
    print("   - The model will start from ground mastery and progress to jump training")
    
    return True

if __name__ == "__main__":
    restore_ground_model()
