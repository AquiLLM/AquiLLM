"""Skill model: per-user markdown blocks merged into the chat system prompt."""
from django.contrib.auth.models import User
from django.db import models


class Skill(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name="skills")
    name = models.CharField(max_length=120)
    body = models.TextField()
    enabled = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        app_label = "apps_skills"
        db_table = "aquillm_skill"
        ordering = ["name"]
        constraints = [
            models.UniqueConstraint(fields=["user", "name"], name="uniq_skill_user_name"),
        ]

    def __str__(self) -> str:
        return f"{self.user.username}: {self.name}"
