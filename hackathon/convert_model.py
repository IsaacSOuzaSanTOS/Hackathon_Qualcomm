#!/usr/bin/env python3
"""
Enhanced AI Hub conversion script for YawnClassifier:
- Exports TorchScript (.pt) for PyTorch    # Path to your trained model checkpoint (.pth)
    pth_path = "models/yawn_model_onnx.pth"  # Use the model we just trained
    if not os.path.exists(pth_path):ployment
- Exports ONNX (.onnx) for onnxruntime-qnn NPU acceleration
- Optionally compiles for Snapdragon X Elite via AI Hub
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.onnx
import os
import sys

try:
    import qai_hub as hub
except ImportError:
    hub = None

# ---- Model definition (must match your training code) ----
class YawnClassifier(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fully_connected1 = nn.Linear(input_dim, 128)  # First fully connected layer
        self.fully_connected2 = nn.Linear(128, 64)         # Second fully connected layer
        self.fully_connected3 = nn.Linear(64, 2)           # Output layer with 2 classes

    def forward(self, x):
        x = F.relu(self.fully_connected1(x))  # ReLU activation after first layer
        x = F.relu(self.fully_connected2(x))  # ReLU activation after second layer
        x = self.fully_connected3(x)          # No activation after final layer (for CrossEntropyLoss)
        return x

# ---- Utility functions ----
def load_model(pth_path, input_dim):
    print(f"📦 Loading model from: {pth_path}")
    model = YawnClassifier(input_dim)
    checkpoint = torch.load(pth_path, map_location='cpu')
    # If checkpoint is a dict with 'model_state_dict', use that
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    elif isinstance(checkpoint, dict):
        model.load_state_dict(checkpoint)
    else:
        model = checkpoint  # If it's already a torch.nn.Module
    model.eval()
    print("✅ Model loaded successfully")
    return model

def export_torchscript(model, example_input, path='yawn_classifier.pt'):
    try:
        print(f"📄 Exporting to TorchScript: {path}")
        traced = torch.jit.trace(model, example_input)
        torch.jit.save(traced, path)
        print(f"✅ TorchScript export successful: {path}")
        return path
    except Exception as e:
        print(f"❌ TorchScript export failed: {e}")
        return None

def export_onnx(model, example_input, path='yawn_classifier.onnx'):
    try:
        print(f"📄 Exporting to ONNX: {path}")
        torch.onnx.export(
            model,
            example_input,
            path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
        )
        # Optional: verify with onnxruntime
        try:
            import onnxruntime as ort
            session = ort.InferenceSession(path, providers=['CPUExecutionProvider'])
            session.run(['output'], {'input': example_input.numpy()})
        except ImportError:
            print("⚠️ onnxruntime not installed, skipping ONNX verification")
        print(f"✅ ONNX export successful: {path}")
        return path
    except Exception as e:
        print(f"❌ ONNX export failed: {e}")
        return None

def compile_for_snapdragon(model_path, input_dim):
    if hub is None:
        print("❌ qai_hub not installed. Install with: pip install qai-hub")
        return None
    print("🔧 Compiling for Snapdragon X Elite...")
    try:
        device = hub.Device("Snapdragon X Elite CRD")
        compile_job = hub.submit_compile_job(
            model=model_path,
            device=device,
            options="--target_runtime qnn_context_binary",
            input_specs={"input": (1, input_dim)}
        )
        print("⏳ Waiting for compilation (this may take several minutes)...")
        compile_job.wait()
        print("✔️ Compilation finished")
        compiled_model = compile_job.get_target_model()
        compiled_path = "yawn_classifier_qnn.bin"
        compiled_model.seek(0)
        with open(compiled_path, 'wb') as f:
            f.write(compiled_model.read())
        print(f"💾 Saved compiled QNN model: {compiled_path}")
        return compiled_path
    except Exception as e:
        print(f"⚠️ Snapdragon compilation failed: {e}")
        return None

def main():
    print("🚀 YawnClassifier Model Conversion Pipeline")
    print("=" * 60)
    # Set input dimension from the mouth landmarks (matches training)
    input_dim = 94  # This matches the training data dimension (X.shape[1])

    # Path to your trained model checkpoint (.pth)
    pth_path = "models/yawn_model_onnx.pth"
    if not os.path.exists(pth_path):
        print(f"❌ Model checkpoint not found: {pth_path}")
        print("   Please export your model's state_dict as .pth using:")
        print("   torch.save(model.state_dict(), 'models/yawn_model_full.pth')")
        print("   If you only have a TorchScript .ts file, retrain or export state_dict.")
        return False

    # Load model
    model = load_model(pth_path, input_dim)
    example_input = torch.randn(1, input_dim)

    # Export TorchScript
    pt_path = export_torchscript(model, example_input)
    # Export ONNX
    onnx_path = export_onnx(model, example_input)

    # Optional: Compile for Snapdragon X Elite
    print("\n" + "-" * 40)
    compile_choice = input("Compile for Snapdragon X Elite? (y/N): ").lower()
    if compile_choice in ['y', 'yes']:
        if pt_path:
            compile_for_snapdragon(pt_path, input_dim)
        else:
            print("⚠️ Cannot compile - TorchScript export failed")

    print(f"\n{'='*60}")
    print("✅ Conversion Pipeline Complete!")
    print(f"{'='*60}")
    print("📋 Generated Files:")
    if pt_path:
        print(f"  • TORCHSCRIPT: {pt_path}")
    if onnx_path:
        print(f"  • ONNX: {onnx_path}")
    print("\n💡 Usage Instructions:")
    if onnx_path:
        print("🔸 For NPU acceleration with onnxruntime-qnn:")
        print("   pip install onnxruntime-qnn")
        print("   python test_onnx_model.py")
    if pt_path:
        print("🔸 For PyTorch deployment:")
        print("   python test_model.py")

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)