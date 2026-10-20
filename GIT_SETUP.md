# Git Repository Setup Guide

This guide helps you initialize and configure your AI Image Detector repository for version control.

## 📋 Quick Setup

```bash
# Navigate to project directory
cd /home/tushar/Desktop/DSBD_project/version1

# Initialize Git repository
git init

# Add all files (respecting .gitignore)
git add .

# Create first commit
git commit -m "Initial commit: AI Image Detector v1.0.0"

# (Optional) Connect to remote repository
git remote add origin <your-repo-url>
git branch -M main
git push -u origin main
```

## 🔍 What Will Be Tracked?

### ✅ Included Files

**Source Code**:
- `src/v1/**/*.py` - All Python source files
- `tests/**/*.py` - Test suite
- `view_db.py` - Utility scripts

**Configuration**:
- `.env.example` - Environment template
- `requirements.txt` - Dependencies
- `pyproject.toml` - Package config
- `.gitignore`, `.gitattributes` - Git config

**Documentation**:
- `README.md` - Main documentation
- `SIMPLE_APP_GUIDE.md` - User guide
- `DASHBOARD_UPDATE.md` - Feature docs
- `BUGFIX_LOCALHOST.md` - Troubleshooting
- `doc/*.pdf` - Research papers

**Dataset Metadata** (small files):
- `datasets/versions.json`
- `datasets/v*/manifest.jsonl`
- `datasets/v*/split_definitions.json`
- `datasets/v*/dataset_report.json`

**Training Metadata**:
- `artifacts/training/*.json` - Training configs and results

### ❌ Excluded Files (in .gitignore)

**Large Binary Files**:
- `resnet50_sd14_best.pt` (~94 MB)
- `clip_probe.joblib` (~7 MB)
- ⚠️ **Action**: Store these in Git LFS or external storage

**Databases**:
- `*.db`, `*.sqlite3` - All SQLite databases
- Contains user data, should not be in version control

**Uploaded Images**:
- `data/uploads/*.png`, `*.jpg`, etc.
- User-submitted content, not for repository

**Generated Artifacts**:
- `artifacts/training/*.pt` - Training checkpoints
- `artifacts/registry/*.pt` - Model registry files

**Environment & Cache**:
- `venv/` - Virtual environment
- `__pycache__/` - Python bytecode
- `.env` - Local secrets
- `.pytest_cache/` - Test cache

## 🔧 Git Configuration

### Set User Information

```bash
git config user.name "Your Name"
git config user.email "your.email@example.com"
```

### Global Ignore Patterns (Optional)

```bash
# Add OS-specific files globally
git config --global core.excludesfile ~/.gitignore_global

# Create global ignore file
cat > ~/.gitignore_global << EOF
.DS_Store
Thumbs.db
*.swp
.vscode/
.idea/
EOF
```

## 📦 Handling Large Model Files

### Option 1: Git LFS (Recommended for GitHub)

```bash
# Install Git LFS
# Ubuntu/Debian:
sudo apt install git-lfs

# macOS:
brew install git-lfs

# Initialize LFS in repo
git lfs install

# Track large files
git lfs track "*.pt"
git lfs track "*.joblib"
git lfs track "*.h5"

# This creates .gitattributes entries
git add .gitattributes

# Add and commit large files
git add resnet50_sd14_best.pt clip_probe.joblib
git commit -m "Add model checkpoints via LFS"
```

**Note**: GitHub LFS has storage limits (1 GB free). Consider alternatives for larger files.

### Option 2: External Storage

Store model files externally and document download instructions:

```bash
# Add to .gitignore (already done)
echo "*.pt" >> .gitignore
echo "*.joblib" >> .gitignore

# Document in README
# Add section: "Download Model Files"
# Provide links to Google Drive, Hugging Face, etc.
```

**Download Script Example**:

```bash
# scripts/download_models.sh
#!/bin/bash
echo "Downloading model checkpoints..."
wget -O resnet50_sd14_best.pt "https://example.com/models/resnet50_sd14_best.pt"
wget -O clip_probe.joblib "https://example.com/models/clip_probe.joblib"
echo "✓ Models downloaded"
```

### Option 3: DVC (Data Version Control)

```bash
# Install DVC
pip install dvc

# Initialize DVC
dvc init

# Add model files to DVC tracking
dvc add resnet50_sd14_best.pt
dvc add clip_probe.joblib

# This creates .dvc files (tracked by Git)
git add resnet50_sd14_best.pt.dvc clip_probe.joblib.dvc .gitignore
git commit -m "Track models with DVC"

# Push to remote storage (S3, Google Drive, etc.)
dvc remote add -d myremote s3://mybucket/dvc-storage
dvc push
```

## 📝 Commit Message Guidelines

Use conventional commit format:

```bash
# Feature
git commit -m "feat: add CLIP model integration"

# Bug fix
git commit -m "fix: resolve database locking issue"

# Documentation
git commit -m "docs: update installation guide"

# Refactor
git commit -m "refactor: reorganize model inference code"

# Tests
git commit -m "test: add dataset builder test cases"

# Chore
git commit -m "chore: update dependencies"
```

## 🌿 Branch Strategy

### Development Workflow

```bash
# Create feature branch
git checkout -b feature/add-model-explainability

# Make changes and commit
git add src/v1/explainability.py
git commit -m "feat: add GradCAM visualization"

# Push to remote
git push origin feature/add-model-explainability

# Create pull request on GitHub/GitLab
```

### Recommended Branches

- `main` - Production-ready code
- `develop` - Integration branch for features
- `feature/*` - New features
- `bugfix/*` - Bug fixes
- `hotfix/*` - Urgent production fixes
- `release/*` - Release preparation

## 🚫 What NOT to Commit

**Never commit**:
- ❌ API keys, tokens, passwords (use `.env`, not in code)
- ❌ User data or uploaded images
- ❌ Large binary files without LFS
- ❌ Virtual environments (`venv/`, `env/`)
- ❌ Database files with personal data
- ❌ IDE-specific config (`.vscode/`, `.idea/`)

**Check before committing**:

```bash
# Review staged files
git status

# Review changes
git diff --cached

# Remove accidentally staged file
git reset HEAD <file>
```

## 🔍 Pre-commit Checks

Create a pre-commit hook:

```bash
# .git/hooks/pre-commit
#!/bin/bash

echo "Running pre-commit checks..."

# Check for secrets
if git diff --cached --name-only | xargs grep -i "api_key\|password\|secret" ; then
    echo "❌ Potential secrets detected!"
    exit 1
fi

# Run tests
pytest -q || exit 1

# Check code format
black --check src/ tests/ || exit 1

echo "✓ Pre-commit checks passed"
```

```bash
chmod +x .git/hooks/pre-commit
```

## 📤 Creating a Release

```bash
# Tag a release
git tag -a v1.0.0 -m "Release version 1.0.0"

# Push tags
git push origin v1.0.0

# Create GitHub release with changelog
# - Go to Releases → Draft new release
# - Select tag v1.0.0
# - Add release notes
# - Attach model files if needed
```

## 🔄 Syncing with Remote

```bash
# Fetch latest changes
git fetch origin

# Merge changes
git merge origin/main

# Or pull (fetch + merge)
git pull origin main

# Push your commits
git push origin main
```

## 🛠️ Useful Git Commands

```bash
# View commit history
git log --oneline --graph --all

# Undo last commit (keep changes)
git reset --soft HEAD~1

# Discard local changes
git checkout -- <file>

# View what would be committed
git diff --cached

# Show files tracked by Git
git ls-files

# Show ignored files
git status --ignored

# Remove file from Git but keep local
git rm --cached <file>

# Check file size in repository
git ls-files -z | xargs -0 du -h | sort -hr | head -20
```

## 🐛 Common Issues

### "File too large" error

```bash
# If you accidentally committed a large file
git rm --cached resnet50_sd14_best.pt
git commit -m "Remove large model file"

# Use Git LFS or external storage instead
```

### Remove sensitive data from history

```bash
# WARNING: Rewrites history, use with caution
git filter-branch --force --index-filter \
  "git rm --cached --ignore-unmatch .env" \
  --prune-empty --tag-name-filter cat -- --all

# Or use BFG Repo-Cleaner (faster)
bfg --delete-files .env
git reflog expire --expire=now --all
git gc --prune=now --aggressive
```

### Merge conflicts

```bash
# View conflicting files
git status

# Edit files to resolve conflicts
# Look for <<<<<<< HEAD markers

# After resolving
git add <resolved-files>
git commit
```

## 📚 Additional Resources

- [Git Documentation](https://git-scm.com/doc)
- [GitHub Guides](https://guides.github.com/)
- [Git LFS](https://git-lfs.github.com/)
- [DVC for ML Projects](https://dvc.org/)
- [Conventional Commits](https://www.conventionalcommits.org/)

---

**Ready to start?** Run the [Quick Setup](#-quick-setup) commands at the top of this guide!
