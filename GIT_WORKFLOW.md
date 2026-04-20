## Key Rules
- fetch/pull before starting work
- Create feature branches for new work
- Write descriptive commit messages
- Rebase instead of merge (cleaner history)
- Never force push to shared branches

## Before Starting Work
```bash
git fetch origin
git pull --rebase origin main
```

## Development Workflow
```bash
# Create feature branch (never commit to main directly)
git checkout -b feature/descriptive-name

# Make changes and commit
git add .
git commit -m "Clear message describing changes"

# Before pushing, sync with latest main
git fetch origin
git rebase origin/main
```

## Pushing Changes
```bash
# Switch to main and merge
git checkout main
git pull origin main
git merge feature/descriptive-name

# Push to remote
git push origin main

# Cleanup
git branch -d feature/descriptive-name
```

## Useful Commands
| Check uncommitted changes | `git status` |
| View commit history | `git log --oneline -10` |
| Undo last commit (keep changes) | `git reset --soft HEAD~1` |
| See unpushed commits | `git log origin/main..main` |
| Sync before work | `git fetch && git pull --rebase` |
| Discard local changes | `git checkout -- <file>` |

```