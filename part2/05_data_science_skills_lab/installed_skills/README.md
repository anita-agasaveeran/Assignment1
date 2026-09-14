# Installed skills (snapshot)

A verbatim copy of `~/.claude/skills/` as it stood when this lab was run: all 46 skills
from the two packs, **with the 16 patches in `../patches/` already applied** to the
bundled scripts.

| Pack | Skills | Source |
|---|---:|---|
| `agent-ml-skills` | 15 | https://github.com/param087/agent-ml-skills |
| `data-analytics-skills` | 31 | https://github.com/nimrodfisher/data-analytics-skills |

This folder is a record, not a live install. Claude Code only loads skills from
`~/.claude/skills/` (global) or `<cwd>/.claude/skills/` (project). To reinstall from
this snapshot:

```bash
cp -R installed_skills/* ~/.claude/skills/
```

To reinstall from upstream instead (and re-hit the bugs the patches fix):

```bash
git clone --depth 1 https://github.com/param087/agent-ml-skills
git clone --depth 1 https://github.com/nimrodfisher/data-analytics-skills
cp -R agent-ml-skills/skills/*                  ~/.claude/skills/
cp -R data-analytics-skills/0[1-6]-*/*          ~/.claude/skills/   # the two 02/ stubs are skipped, see ../README.md
cd ~/.claude/skills && for p in <lab>/patches/*.patch; do patch -p0 < "$p"; done
```
