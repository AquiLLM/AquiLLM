"""
Management command: seed_feedback

Creates realistic fake feedback data for dashboard development and testing.
Safe to run multiple times — it checks for existing seed users first.

Usage:
    python manage.py seed_feedback
    python manage.py seed_feedback --clear     # wipe seed data and re-create
    python manage.py seed_feedback --count 20  # messages per user (default 18)
"""
from __future__ import annotations

import random
from datetime import timedelta

from django.contrib.auth import get_user_model
from django.core.management.base import BaseCommand
from django.utils import timezone

from apps.chat.models import Message
from apps.chat.models.conversation import WSConversation

User = get_user_model()

# ---------------------------------------------------------------------------
# seed constants
# ---------------------------------------------------------------------------

SEED_USERNAMES = ["alice_seed", "bob_seed", "carol_seed", "dave_seed"]

MODELS = [
    "claude-3-5-sonnet-20241022",
    "claude-3-opus-20240229",
    "claude-3-haiku-20240307",
    "claude-3-5-haiku-20241022",
]

TOOL_NAMES = [
    None, None, None, None,   # most messages have no tool
    "search_arxiv",
    "retrieve_chunks",
    "web_search",
    "calculator",
]

STOP_REASONS = ["end_turn", "max_tokens", "stop_sequence"]

USER_QUESTIONS = [
    "Can you explain how transformers work in NLP?",
    "What is the difference between precision and recall?",
    "How do I fine-tune a language model on my own dataset?",
    "Explain gradient descent intuitively.",
    "What are the main challenges in multimodal AI?",
    "How does retrieval augmented generation work?",
    "What is the attention mechanism and why does it matter?",
    "Compare BERT and GPT architectures.",
    "What is constitutional AI?",
    "How do I evaluate a language model's performance?",
    "What is the role of temperature in language model sampling?",
    "Explain the concept of embeddings.",
    "What are the limitations of current large language models?",
    "How does reinforcement learning from human feedback work?",
    "What is chain-of-thought prompting?",
    "Explain the difference between zero-shot and few-shot learning.",
    "What is quantization and how does it affect model performance?",
    "How do mixture-of-experts models work?",
]

ASSISTANT_RESPONSES = [
    "Transformers are a type of neural network architecture that uses self-attention mechanisms "
    "to process sequential data. Unlike RNNs, transformers process all tokens in parallel, "
    "making them highly efficient on modern hardware. The key innovation is the multi-head "
    "attention mechanism which allows the model to attend to different positions simultaneously. "
    "This enables transformers to capture long-range dependencies that RNNs struggle with. "
    "The original transformer paper 'Attention is All You Need' introduced this architecture "
    "in 2017 and it has since become the foundation for virtually all modern language models.",

    "Precision measures what fraction of your positive predictions were actually correct. "
    "Recall measures what fraction of all actual positives you correctly identified. "
    "A high precision model rarely cries wolf — when it says positive, it usually is. "
    "A high recall model catches almost everything positive, but may have many false alarms. "
    "The F1 score is the harmonic mean of precision and recall, balancing both concerns. "
    "Which metric matters more depends entirely on your use case and the cost of each error type.",

    "Fine-tuning a language model involves continuing its training on your domain-specific data. "
    "The key steps are: prepare a high quality dataset in the expected format, choose whether "
    "to do full fine-tuning or parameter-efficient methods like LoRA, set a learning rate "
    "significantly lower than pretraining (typically 1e-5 to 1e-4), monitor for overfitting, "
    "and evaluate on a held-out validation set. LoRA is often preferred as it requires far "
    "less compute and memory while achieving comparable results to full fine-tuning.",

    "Gradient descent is an optimization algorithm that iteratively moves in the direction "
    "that reduces loss. Imagine you are blindfolded on a hilly landscape and want to find "
    "the lowest valley. You feel the slope under your feet (the gradient) and take a small "
    "step downhill. Repeat this many times and you will eventually reach a low point. "
    "The learning rate controls how large each step is. Too large and you overshoot. "
    "Too small and convergence is painfully slow.",

    "Multimodal AI faces several core challenges. First, aligning representations across "
    "modalities so that the model understands that an image of a cat and the word cat refer "
    "to the same concept. Second, handling the different data densities — a single image "
    "contains vastly more raw information than a sentence. Third, collecting and curating "
    "paired multimodal training data is expensive and time-consuming. Fourth, evaluation "
    "is harder because there is no single ground truth for tasks like image captioning.",

    "RAG combines a retrieval system with a generative model. When a query arrives, "
    "the retrieval component searches a knowledge base for relevant documents using "
    "dense vector similarity or sparse keyword matching. These retrieved chunks are "
    "prepended to the prompt as context, giving the generative model grounding in "
    "specific factual content. This dramatically reduces hallucination for factual "
    "questions and allows the knowledge base to be updated without retraining.",
]

FEEDBACK_TEXTS = [
    "This answer was extremely helpful, explained everything clearly.",
    "Good response but could use more detail on the mathematical foundations.",
    "Not quite what I was looking for — too high level.",
    "Perfect, exactly what I needed for my research.",
    "The response was a bit too long and wandered off topic near the end.",
    "Very accurate and the citations were useful.",
    "Missed the point of my question entirely.",
    "Solid answer. Would have liked a concrete code example.",
    "Great explanation of the intuition, but the technical details were off.",
    "Clear, concise, and directly answered my question.",
    "The analogy used was very helpful for understanding this concept.",
    "Too much jargon without explanation for a beginner audience.",
    "Excellent breakdown of the trade-offs involved.",
    "The answer contradicted something I read elsewhere, needs verification.",
    "Really appreciated the step-by-step structure.",
    "",     # empty string = no feedback text, row still counts if rated
    "",
    None,   # null = no feedback text
    None,
    None,
]

RATINGS = [1, 2, 3, 3, 4, 4, 4, 5, 5, 5, None, None]  # weighted toward 4-5


# ---------------------------------------------------------------------------
# command
# ---------------------------------------------------------------------------

class Command(BaseCommand):
    help = "seed the database with fake feedback data for dashboard development"

    def add_arguments(self, parser):
        parser.add_argument(
            "--clear",
            action="store_true",
            help="delete all data owned by seed users before re-seeding",
        )
        parser.add_argument(
            "--count",
            type=int,
            default=18,
            help="number of assistant messages (with feedback) per user (default 18)",
        )

    def handle(self, *args, **options):
        clear = options["clear"]
        msg_count = options["count"]

        if clear:
            self._clear_seed_data()

        users = self._ensure_seed_users()
        total_created = 0

        for user in users:
            total_created += self._seed_user(user, msg_count)

        # verify via the actual dataset queryset so we confirm the filter works
        from apps.platform_admin.services.feedback_dataset import feedback_dataset_queryset
        total_feedback = feedback_dataset_queryset().count()

        self.stdout.write(
            self.style.SUCCESS(
                f"seeding complete — created {total_created} assistant messages, "
                f"{total_feedback} are feedback-bearing and visible in the dashboard"
            )
        )

    # -----------------------------------------------------------------------
    # private helpers
    # -----------------------------------------------------------------------

    def _clear_seed_data(self):
        """delete all conversations and messages owned by seed users"""
        deleted_convos = 0
        deleted_msgs = 0
        for username in SEED_USERNAMES:
            try:
                user = User.objects.get(username=username)
                convos = WSConversation.objects.filter(owner=user)
                for convo in convos:
                    n, _ = Message.objects.filter(conversation=convo).delete()
                    deleted_msgs += n
                n, _ = convos.delete()
                deleted_convos += n
            except User.DoesNotExist:
                pass
        self.stdout.write(
            f"cleared {deleted_convos} conversations and {deleted_msgs} messages"
        )

    def _ensure_seed_users(self) -> list:
        """get or create the seed users, return the list"""
        users = []
        for username in SEED_USERNAMES:
            user, created = User.objects.get_or_create(
                username=username,
                defaults={
                    "email": f"{username}@seed.example.com",
                    "first_name": username.split("_")[0].capitalize(),
                },
            )
            if created:
                user.set_password("seedpass123!")
                user.save()
                self.stdout.write(f"created user: {username}")
            else:
                self.stdout.write(f"found existing user: {username}")
            users.append(user)
        return users

    def _seed_user(self, user, msg_count: int) -> int:
        """create conversations and feedback messages for one user, return count of assistant messages"""
        now = timezone.now()
        convo_count = max(1, msg_count // 6)
        assistant_count = 0

        for convo_num in range(convo_count):
            convo = WSConversation.objects.create(
                owner=user,
                name=f"{user.first_name} — topic {convo_num + 1}",
            )

            msgs_this_convo = msg_count // convo_count
            for i in range(msgs_this_convo):
                seq_base = i * 2

                # user turn
                Message.objects.create(
                    conversation=convo,
                    role="user",
                    content=random.choice(USER_QUESTIONS),
                    sequence_number=seq_base,
                )

                # assistant turn with feedback
                rating = random.choice(RATINGS)
                feedback_text = random.choice(FEEDBACK_TEXTS)

                # ensure at least rating or non-empty text so the row is feedback-bearing
                # (if both are null/empty, force a rating so the dashboard sees it)
                if rating is None and (feedback_text is None or not feedback_text.strip()):
                    rating = random.choice([3, 4, 5])

                submitted_at = None
                if rating is not None or (feedback_text and feedback_text.strip()):
                    days_ago = random.randint(0, 90)
                    hours_ago = random.randint(0, 23)
                    submitted_at = now - timedelta(days=days_ago, hours=hours_ago)

                Message.objects.create(
                    conversation=convo,
                    role="assistant",
                    content=random.choice(ASSISTANT_RESPONSES),
                    sequence_number=seq_base + 1,
                    model=random.choice(MODELS),
                    tool_call_name=random.choice(TOOL_NAMES),
                    stop_reason=random.choice(STOP_REASONS),
                    usage=random.randint(300, 6000),
                    rating=rating,
                    feedback_text=feedback_text if feedback_text else None,
                    feedback_submitted_at=submitted_at,
                )
                assistant_count += 1

        return assistant_count