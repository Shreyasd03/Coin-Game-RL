#!/usr/bin/env python3
"""
Master Model Management System
- List available master models
- Restore specific master models
- Create new master models from current training
- Delete old master models
"""

import os
import shutil
from datetime import datetime

def list_master_models():
    """List all available master models."""
    if not os.path.exists("masterModels"):
        print("❌ No masterModels directory found")
        return []
    
    master_files = [f for f in os.listdir("masterModels") if f.endswith('.zip')]
    if not master_files:
        print("📁 No master models found")
        return []
    
    print("📁 Available Master Models:")
    print("="*50)
    for i, model_file in enumerate(master_files, 1):
        model_name = model_file.replace('.zip', '')
        model_path = f"masterModels/{model_file}"
        file_size = os.path.getsize(model_path) / (1024*1024)  # Size in MB
        mod_time = datetime.fromtimestamp(os.path.getmtime(model_path))
        print(f"{i:2d}. {model_name}")
        print(f"    📅 Created: {mod_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"    📦 Size: {file_size:.1f} MB")
        print()
    
    return master_files

def restore_master_model():
    """Restore a specific master model to main model files."""
    master_files = list_master_models()
    if not master_files:
        return False
    
    print("🎯 Select master model to restore:")
    while True:
        try:
            choice = int(input("Enter model number: ")) - 1
            if 0 <= choice < len(master_files):
                break
            else:
                print(f"❌ Invalid choice. Please enter 1-{len(master_files)}")
        except (ValueError, KeyboardInterrupt):
            print("❌ Invalid input or cancelled")
            return False
    
    selected_model = master_files[choice]
    model_name = selected_model.replace('.zip', '')
    
    # Paths
    master_model_path = f"masterModels/{selected_model}"
    master_vecnormalize_path = f"masterModels/vecnormalize_{model_name.split('_')[-1]}.pkl"
    main_model_path = "models/ppo_coin_final.zip"
    main_vecnormalize_path = "models/vecnormalize.pkl"
    
    # Check if master model exists
    if not os.path.exists(master_model_path):
        print(f"❌ Master model not found: {master_model_path}")
        return False
    
    # Backup current model if it exists
    if os.path.exists(main_model_path):
        backup_model = f"models/ppo_coin_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
        shutil.copy(main_model_path, backup_model)
        print(f"📦 Backed up current model to: {backup_model}")
    
    if os.path.exists(main_vecnormalize_path):
        backup_vecnormalize = f"models/vecnormalize_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        shutil.copy(main_vecnormalize_path, backup_vecnormalize)
        print(f"📦 Backed up current VecNormalize to: {backup_vecnormalize}")
    
    # Restore master model
    shutil.copy(master_model_path, main_model_path)
    if os.path.exists(master_vecnormalize_path):
        shutil.copy(master_vecnormalize_path, main_vecnormalize_path)
        print(f"✅ Restored master model: {model_name}")
        print(f"✅ Restored VecNormalize: vecnormalize_{model_name.split('_')[-1]}.pkl")
    else:
        print(f"⚠️  Master model restored but VecNormalize not found: {master_vecnormalize_path}")
        print("   You may need to retrain to get proper normalization stats")
    
    print("\n🎯 You can now:")
    print("   - Run visual evaluation: python -m rl.visualEval")
    print("   - Continue training: python -m rl.trainPPO")
    
    return True

def create_master_from_current():
    """Create a new master model from current training model."""
    current_model = "models/ppo_coin_final.zip"
    current_vecnormalize = "models/vecnormalize.pkl"
    
    if not os.path.exists(current_model):
        print("❌ No current model found. Train a model first.")
        return False
    
    print("🎯 Create Master Model from Current Training")
    print("="*50)
    print("Available models in models/ folder:")
    model_files = [f for f in os.listdir("models") if f.endswith('.zip')]
    for i, model_file in enumerate(model_files, 1):
        print(f"  {i}. {model_file}")
    
    # Let user choose which model to use as master
    while True:
        try:
            choice = input(f"Select model to use as master (1-{len(model_files)}) or press Enter for final: ").strip()
            if choice == "":
                source_model = current_model
                source_vecnormalize = current_vecnormalize
                break
            else:
                choice_idx = int(choice) - 1
                if 0 <= choice_idx < len(model_files):
                    source_model = f"models/{model_files[choice_idx]}"
                    # Try to find corresponding VecNormalize
                    model_name = model_files[choice_idx].replace('.zip', '')
                    source_vecnormalize = f"models/vecnormalize_{model_name.split('_')[-1]}.pkl"
                    if not os.path.exists(source_vecnormalize):
                        source_vecnormalize = current_vecnormalize  # Fallback to current
                    break
                else:
                    print(f"❌ Invalid choice. Please enter 1-{len(model_files)} or press Enter")
        except (ValueError, KeyboardInterrupt):
            print("❌ Invalid input or cancelled")
            return False
    
    master_name = input("Enter master model name (e.g., 'ground_master', 'jump_master'): ").strip()
    if not master_name:
        print("❌ Master name cannot be empty")
        return False
    
    # Create masterModels directory if it doesn't exist
    os.makedirs("masterModels", exist_ok=True)
    
    # Save master model
    master_model_path = f"masterModels/ppo_coin_{master_name}.zip"
    master_vecnormalize_path = f"masterModels/vecnormalize_{master_name}.pkl"
    
    shutil.copy(source_model, master_model_path)
    if os.path.exists(source_vecnormalize):
        shutil.copy(source_vecnormalize, master_vecnormalize_path)
        print(f"✅ Master model created: {master_model_path}")
        print(f"✅ Master VecNormalize created: {master_vecnormalize_path}")
    else:
        print(f"✅ Master model created: {master_model_path}")
        print("⚠️  VecNormalize not found - you may need to retrain for proper normalization")
    
    return True

def delete_master_model():
    """Delete a master model."""
    master_files = list_master_models()
    if not master_files:
        return False
    
    print("🗑️  Select master model to delete:")
    while True:
        try:
            choice = int(input("Enter model number: ")) - 1
            if 0 <= choice < len(master_files):
                break
            else:
                print(f"❌ Invalid choice. Please enter 1-{len(master_files)}")
        except (ValueError, KeyboardInterrupt):
            print("❌ Invalid input or cancelled")
            return False
    
    selected_model = master_files[choice]
    model_name = selected_model.replace('.zip', '')
    
    # Confirm deletion
    confirm = input(f"⚠️  Are you sure you want to delete '{model_name}'? (y/N): ").strip().lower()
    if confirm in ['y', 'yes']:
        master_model_path = f"masterModels/{selected_model}"
        master_vecnormalize_path = f"masterModels/vecnormalize_{model_name.split('_')[-1]}.pkl"
        
        os.remove(master_model_path)
        if os.path.exists(master_vecnormalize_path):
            os.remove(master_vecnormalize_path)
        
        print(f"✅ Deleted master model: {model_name}")
        return True
    else:
        print("❌ Deletion cancelled")
        return False

def main():
    """Main menu for master model management."""
    while True:
        print("\n" + "="*60)
        print("🎯 MASTER MODEL MANAGEMENT")
        print("="*60)
        print("1. 📁 List Master Models")
        print("2. 🔄 Restore Master Model")
        print("3. 💾 Create Master from Current")
        print("4. 🗑️  Delete Master Model")
        print("5. 🚪 Exit")
        print("="*60)
        
        try:
            choice = input("Enter your choice (1-5): ").strip()
            
            if choice == '1':
                list_master_models()
            elif choice == '2':
                restore_master_model()
            elif choice == '3':
                create_master_from_current()
            elif choice == '4':
                delete_master_model()
            elif choice == '5':
                print("👋 Goodbye!")
                break
            else:
                print("❌ Invalid choice. Please enter 1-5.")
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break

if __name__ == "__main__":
    main()
