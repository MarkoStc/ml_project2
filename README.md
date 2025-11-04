
# 🧠 Team Git Basics (Short & Simple)

## 🏁 One-time setup (per developer)

```bash
# Clone the repo
git clone https://github.com/<org-or-username>/ml_project2.git
cd ml_project2
````

### 🔧 Auto-strip Jupyter notebook outputs

```bash
pip install nbstripout
nbstripout --install --attributes .gitattributes
```

✅ This ensures any `*.ipynb` you commit gets its **outputs removed automatically**.

> 💡 **Windows tip:**
> Run `git config core.autocrlf true` to silence CRLF/LF warnings.

---

## 🌿 Create your branch (first time only)

```bash
git checkout -b feature/marko-dev
git push -u origin feature/marko-dev
```

> `-u` lets you use plain `git push` / `git pull` next time without specifying the branch.

---

## 🔁 Daily workflow (no syncing with main here)

```bash
# Work on your branch
git add .
git commit -m "Describe your change"
git push    # pushes to feature/marko-dev
```

That’s it for day-to-day work.

---

## 🚀 End of feature (sync with main + PR)

### 1️⃣ Update your branch with main

```bash
git fetch origin
git rebase origin/main      # or: git merge origin/main
```

### 2️⃣ If you rebased, update remote branch

```bash
git push --force-with-lease
```

### 3️⃣ Open a Pull Request

* **base:** `main`
* **compare:** `feature/marko-dev`
* Get a review → **Squash & Merge** (recommended)

### 4️⃣ After merge

```bash
git checkout main
git pull
git checkout -b feature/<next-task>
```

---

## 🧹 If notebooks with outputs were already committed (one-time cleanup)

### Quick clean (any OS)

```bash
nbstripout --force
git add .
git commit -m "Strip notebook outputs"
git push
```

### PowerShell (shows progress)

```powershell
Get-ChildItem . -Filter *.ipynb -Recurse | ForEach-Object {
  Write-Host "🧹 $($_.FullName)"
  python -m nbstripout "$($_.FullName)"
}
git add .
git commit -m "Strip notebook outputs"
git push
```

---

## 🧭 Quick reference

| Task                            | Command                                                                         |
| ------------------------------- | ------------------------------------------------------------------------------- |
| Create and track branch         | `git checkout -b feature/marko-dev`<br>`git push -u origin feature/marko-dev`   |
| Daily push                      | `git add . && git commit -m "msg" && git push`                                  |
| Sync with main (end of feature) | `git fetch origin`<br>`git rebase origin/main`<br>`git push --force-with-lease` |
| Open PR                         | compare `feature/marko-dev` → base `main`                                       |

---

👨‍💻 **Team rule:**
Work only on your branch during development.
We sync with `main` *only at the end* before merging via Pull Request.

````


