# 🎮 Google Colab Setup Instructions for BalatroNN

## 🚀 Quick Start (5 Minutes)

### Step 1: Open Colab
1. Go to [colab.research.google.com](https://colab.research.google.com)
2. Sign in with your Google account

### Step 2: Upload Notebook
**Option A: From GitHub (Easiest)**
1. File → Open notebook
2. Click "GitHub" tab
3. Enter: `sksksksksksksk/balatroNN`
4. Select `a100_training.ipynb`
5. Click to open

**Option B: Upload File**
1. Download `a100_training.ipynb` from [github.com/sksksksksksksk/balatroNN](https://github.com/sksksksksksksk/balatroNN)
2. File → Upload notebook
3. Select the downloaded file

### Step 3: Set A100 GPU
1. **Runtime → Change runtime type**
2. **Hardware accelerator:** GPU
3. **GPU type:** A100
4. Click **Save**

⚠️ **Important**: A100 requires **Colab Pro+** ($50/month)
- Without Pro+: Use T4 (free) with `configs/colab.yaml` instead
- With Pro: A100 may be available depending on demand

### Step 4: Run Everything
1. **Runtime → Run all** (or press Ctrl+F9)
2. Wait 2-3 minutes for setup
3. Training begins automatically!

**That's it!** ✨

---

## ⏱️ Training Duration

- **Setup**: 2-3 minutes
- **Training**: 12-16 hours
- **Total**: ~13-17 hours

---

## 💾 Keeping Colab Connected

Colab disconnects after **~12 hours** on free tier, **24 hours** on Pro+.

### For 12-16 Hour Training:

#### Option 1: Colab Pro+ (Recommended)
- $50/month subscription
- 24-hour runtime (covers full training)
- Background execution
- Priority A100 access
- **This is the best option**

#### Option 2: Auto-Reconnect Script
Add this cell at the beginning and run it:

```javascript
%%javascript
function ClickConnect(){
  console.log("Keep-alive clicked");
  document.querySelector("#top-toolbar > colab-connect-button").shadowRoot.querySelector("#connect").click()
}
setInterval(ClickConnect, 60000);  // Click every minute
```

This keeps the browser "active" but may not work for 12+ hours.

#### Option 3: Check-In Method
- Set a timer for every 4-6 hours
- Check if Colab is still connected
- If disconnected, it will resume from last checkpoint
- Re-run the training cell

---

## 💰 Cost Breakdown

| Plan | Cost | Runtime | A100 Access | Best For |
|------|------|---------|-------------|----------|
| **Free** | $0 | ~12h | ❌ T4 only | Quick tests |
| **Pro** | $10/mo | ~24h | ⚠️ Limited | Casual use |
| **Pro+** | $50/mo | 24h | ✅ Priority | **Training** ⭐ |

**For this training**: Pro+ is highly recommended ($50 well spent)

---

## 🔄 If Training Gets Interrupted

Don't worry! The notebook has automatic checkpointing.

### To Resume:
1. Reconnect to runtime
2. Run the **setup cell** (Cell 2)
3. Run the **training cell** (Cell 4)
4. It automatically resumes from the last checkpoint!

Checkpoints are saved every ~30 minutes in `/content/balatroNN/checkpoints/a100/`

---

## 📊 Monitoring Training

### Option 1: TensorBoard (Built-in)
Run Cell 6 while training runs to see live graphs:
- Loss curves
- Reward progression
- Training metrics

### Option 2: Check Logs
```python
!tail -f logs/a100_training/*/events*
```

---

## 🗂️ Downloading Your Trained Model

After training completes:

### Option 1: Direct Download
```python
from google.colab import files
files.download('checkpoints/a100/*/final_model.pt')
```

### Option 2: Google Drive
Add this cell after training:
```python
from google.colab import drive
drive.mount('/content/drive')

!cp -r checkpoints/a100 /content/drive/MyDrive/balatroNN_checkpoints/
```

### Option 3: GitHub (Best for sharing)
```python
!git config --global user.email "you@example.com"
!git config --global user.name "Your Name"
!git add checkpoints/a100/*/final_model.pt
!git commit -m "Add trained model"
!git push
```

---

## 🆘 Troubleshooting

### "A100 not available"
- A100 requires Colab Pro+ ($50/month)
- Alternative: Use T4 (free) with different config:
  ```python
  !python train.py --config configs/colab.yaml
  ```
  Will be slower but works on free tier

### "Runtime disconnected"
- Normal after 12h on free tier
- Resume by re-running training cell
- Or upgrade to Pro+ for 24h runtime

### "Out of memory"
Edit `configs/a100.yaml`:
```yaml
training:
  batch_size: 512  # Reduce from 768
```

### "CUDA error"
- Restart runtime: Runtime → Restart runtime
- Re-run all cells

### "Repository clone failed"
- Check internet connection
- Verify repo exists: [github.com/sksksksksksksk/balatroNN](https://github.com/sksksksksksksk/balatroNN)
- Try cloning manually:
  ```python
  !git clone https://github.com/sksksksksksksk/balatroNN.git
  ```

### "Training seems stuck"
Check GPU usage:
```python
!nvidia-smi
```
Should show ~90-100% GPU utilization.

---

## 📝 Configuration Tips

### Faster Training (Lower Quality)
```yaml
# configs/a100.yaml
training:
  total_timesteps: 10_000_000  # 10M instead of 50M
```
Training time: ~3-4 hours instead of 12-16

### Smaller Model (Less Memory)
```yaml
model:
  d_model: 256  # Instead of 384
  num_layers: 6  # Instead of 8
```

### Disable Curriculum Learning
```yaml
curriculum:
  enabled: false
```

---

## ✨ Pro Tips

1. **Start small**: Test with 1M timesteps first (configs/quick_test.yaml)
2. **Enable W&B**: Remote monitoring is worth it
3. **Save to Drive**: Auto-backup checkpoints to Google Drive
4. **Pro+ is worth it**: For 12-16h training, the $50 is justified
5. **Run overnight**: Start evening, wake up to trained model

---

## 📚 Additional Resources

- **Main README**: [github.com/sksksksksksksk/balatroNN](https://github.com/sksksksksksksk/balatroNN)
- **Detailed Guide**: `A100_TRAINING_GUIDE.md`
- **Quick Reference**: `ONE_CLICK_TRAINING.md`

---

## 🎯 Expected Results

After 50M timesteps (12-16 hours):
- **Ante 1-3**: Consistent wins (>90%)
- **Ante 4-5**: ~70% win rate
- **Ante 6+**: Learning, ~30-40% wins

Your AI will understand:
- Hand selection strategy
- Joker synergies
- Discard optimization  
- Shop economics
- Blind mechanics

---

## 🎉 Quick Checklist

Before running, verify:
- [ ] Colab Pro+ subscription active
- [ ] A100 GPU selected in runtime
- [ ] Notebook uploaded/loaded
- [ ] Ready to commit 12-16 hours

Then:
- [ ] Click "Run all"
- [ ] Wait ~2 minutes for setup
- [ ] Leave it running
- [ ] Come back in 12-16 hours
- [ ] Enjoy your trained Balatro AI!

---

**Need help?** Check the troubleshooting section or open an issue on GitHub!

**Ready?** Open the notebook and click "Run all"! 🚀

