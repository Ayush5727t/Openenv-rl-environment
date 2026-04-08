# Deployment Guide for OpenEnv Environment

## 📦 Hugging Face Spaces Deployment

### Prerequisites

1. A Hugging Face account
2. Git installed locally
3. The OpenEnv environment code

### Step-by-Step Deployment

#### 1. Create a New Space

1. Go to https://huggingface.co/spaces
2. Click "Create new Space"
3. Choose:
   - **Name**: `openenv-environment` (or your preferred name)
   - **License**: MIT
   - **SDK**: Docker
   - **Visibility**: Public

#### 2. Configure Space

Create a `README.md` in your Space with metadata:

```markdown
---
title: OpenEnv Environment
emoji: 🤖
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
tags:
  - openenv
  - reinforcement-learning
  - ai-agents
---

# OpenEnv Environment

A complete environment for AI agents with file management, text processing, and data processing tasks.
```

#### 3. Push Code to Space

```bash
# Clone your Space
git clone https://huggingface.co/spaces/YOUR_USERNAME/openenv-environment
cd openenv-environment

# Copy all project files
cp -r /path/to/openenv/* .

# Add and commit
git add .
git commit -m "Initial deployment"

# Push to Space
git push
```

#### 4. Verify Deployment

1. Wait for the Space to build (check the "Build" tab)
2. Once built, the app will be available at your Space URL
3. Test the interface by initializing an environment and running actions

### 🐳 Local Docker Testing

Before deploying, test locally:

```bash
# Build image
docker build -t openenv:test .

# Run container
docker run -p 7860:7860 openenv:test

# Open browser
open http://localhost:7860
```

### 🔧 Configuration Options

#### Environment Variables

None required for basic deployment. Optional variables:

- `GRADIO_SERVER_NAME`: Default "0.0.0.0"
- `GRADIO_SERVER_PORT`: Default 7860

#### Custom Domain

To use a custom domain:

1. Go to Space Settings
2. Add custom domain
3. Follow DNS configuration instructions

### 📊 Monitoring

View logs in the "Logs" tab of your Space to monitor:
- Build process
- Runtime errors
- User interactions

### 🔄 Updating Your Deployment

```bash
# Make changes locally
git pull
# ... make your changes ...

# Commit and push
git add .
git commit -m "Update: description of changes"
git push

# Space will automatically rebuild
```

### 🚨 Troubleshooting

#### Build Failures

1. Check `Dockerfile` syntax
2. Verify all dependencies in `requirements.txt`
3. Check build logs for specific errors

#### Runtime Issues

1. Check container logs
2. Verify port 7860 is exposed
3. Ensure all imports work correctly

#### Memory Issues

If the Space runs out of memory:
1. Consider using a smaller base image
2. Optimize dependencies
3. Upgrade to a larger Space tier

### 📈 Scaling

For production use:
- Upgrade to a paid Space tier for more resources
- Enable auto-scaling if available
- Consider using HF Inference Endpoints for high traffic

## 🐙 Alternative: GitHub + Hugging Face Integration

1. Push code to GitHub
2. Connect GitHub repository to HF Space
3. Auto-deploy on push to main branch

### GitHub Actions Workflow

```yaml
name: Deploy to HF Space

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Push to HF Space
        env:
          HF_TOKEN: ${{ secrets.HF_TOKEN }}
        run: |
          git remote add space https://huggingface.co/spaces/YOUR_USERNAME/openenv-environment
          git push space main
```

## 📋 Deployment Checklist

- [ ] Code tested locally
- [ ] Docker image builds successfully
- [ ] All tests passing
- [ ] README.md with Space metadata
- [ ] requirements.txt up to date
- [ ] Dockerfile configured correctly
- [ ] app.py runs without errors
- [ ] Push to Hugging Face Space
- [ ] Verify Space builds successfully
- [ ] Test web interface
- [ ] Update Space description/documentation

## 🎉 Post-Deployment

After successful deployment:

1. **Share Your Space**: Add to HF collections, share on social media
2. **Documentation**: Update README with Space URL
3. **Examples**: Add example notebooks/colab links
4. **Community**: Engage with users in discussions
5. **Monitor**: Watch analytics and user feedback

## 📞 Support

- **Hugging Face Docs**: https://huggingface.co/docs/hub/spaces
- **Community Forum**: https://discuss.huggingface.co/
- **Discord**: Join HF Discord server

---

For more information, visit the [Hugging Face Spaces documentation](https://huggingface.co/docs/hub/spaces).
