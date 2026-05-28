"""User-authored markdown skills.

Per-user, DB-backed markdown blocks that are merged into each user's chat
system prompt at runtime. Complements the filesystem-loaded markdown skills
in `lib.skills.markdown` (which remain global / ops-managed).
"""

default_app_config = "apps.skills.apps.SkillsConfig"
