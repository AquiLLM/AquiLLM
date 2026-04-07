"""
seed_data.py — Populate a dev instance with example PDFs, conversations, and feedback.

Downloads 7 short arxiv papers (ML/CV), creates two collections, three seed users,
and 15 conversations with a mix of star ratings and written feedback text.

Usage
-----
On dev2 / production-mirrored environments (uses .env directly):
    docker compose -f deploy/compose/base.yml -f deploy/compose/production.yml exec web python manage.py seed_data

On local no-GPU dev (requires --env-file so YAML variable substitution resolves):
    docker compose --env-file .env -f deploy/compose/no_gpu_dev.yml -f deploy/compose/observability.yml exec web python manage.py seed_data

Options:
    --force     Delete all existing seed data and recreate from scratch

Notes:
- Requires internet access to download PDFs from arxiv.org at run time.
- PDF chunking/embedding runs async via Celery after the script finishes.
  Documents will appear in the ingestion dashboard and become searchable
  once the worker completes (typically under a minute per paper).
- Seed user credentials: seed_alice / seed_bob / seed_carol, password: seedpassword123
"""

import io
import random
import urllib.request
from datetime import timedelta

from django.contrib.auth.models import User
from django.core.files.base import ContentFile
from django.core.management.base import BaseCommand
from django.utils import timezone
from pypdf import PdfReader

from aquillm.models import (
    Collection,
    CollectionPermission,
    Message,
    PDFDocument,
    WSConversation,
)

# ── Seed identity ─────────────────────────────────────────────────────────────

SEED_TAG = "seed_"  # all seed usernames are prefixed with this

SEED_USERS = [
    {"username": "seed_alice", "email": "alice@seed.dev", "first_name": "Alice", "last_name": "Seed"},
    {"username": "seed_bob",   "email": "bob@seed.dev",   "first_name": "Bob",   "last_name": "Seed"},
    {"username": "seed_carol", "email": "carol@seed.dev", "first_name": "Carol", "last_name": "Seed"},
]

# ── Collections ───────────────────────────────────────────────────────────────

COLLECTION_DEFS = [
    {
        "name": "ML & NLP Foundations",
        "owner": "seed_alice",
        "editors": ["seed_bob"],
        "viewers": ["seed_carol"],
    },
    {
        "name": "Computer Vision Papers",
        "owner": "seed_bob",
        "editors": [],
        "viewers": ["seed_alice", "seed_carol"],
    },
]

# ── ArXiv papers ──────────────────────────────────────────────────────────────
# Deliberately short papers only 

ARXIV_PAPERS = [
    # ML & NLP Foundations
    {
        "arxiv_id": "1706.03762",
        "title": "Attention Is All You Need",
        "collection": "ML & NLP Foundations",
        "uploader": "seed_alice",
    },
    {
        "arxiv_id": "1301.3666",
        "title": "Efficient Estimation of Word Representations in Vector Space",
        "collection": "ML & NLP Foundations",
        "uploader": "seed_alice",
    },
    {
        "arxiv_id": "1810.04805",
        "title": "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
        "collection": "ML & NLP Foundations",
        "uploader": "seed_bob",
    },
    {
        "arxiv_id": "1312.6114",
        "title": "Auto-Encoding Variational Bayes",
        "collection": "ML & NLP Foundations",
        "uploader": "seed_bob",
    },
    # Computer Vision Papers
    {
        "arxiv_id": "2010.11929",
        "title": "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale",
        "collection": "Computer Vision Papers",
        "uploader": "seed_bob",
    },
    {
        "arxiv_id": "1512.03385",
        "title": "Deep Residual Learning for Image Recognition",
        "collection": "Computer Vision Papers",
        "uploader": "seed_bob",
    },
    {
        "arxiv_id": "2103.00020",
        "title": "Learning Transferable Visual Models From Natural Language Supervision",
        "collection": "Computer Vision Papers",
        "uploader": "seed_carol",
    },
]

# ── Conversations ─────────────────────────────────────────────────────────────
# Each entry is a dict with: user, name, messages
# Each message: role, content, and for assistant messages: model, stop_reason,
#   usage, and optionally rating (1-5) and feedback_text.

MODEL = "gpt-4o"

CONVERSATIONS = [
    {
        "user": "seed_alice",
        "name": "Attention Mechanism Explained",
        "messages": [
            {
                "role": "user",
                "content": "Can you explain how the attention mechanism works in the Transformer paper?",
            },
            {
                "role": "assistant",
                "content": (
                    "The attention mechanism computes a weighted sum of values, where each weight reflects "
                    "how relevant a given key is to the current query.\n\n"
                    "Formally: Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) · V\n\n"
                    "The division by sqrt(d_k) keeps dot-product magnitudes stable as dimension grows, "
                    "preventing the softmax from saturating into near-zero gradients.\n\n"
                    "The paper stacks this into Multi-Head Attention: the same operation runs h times "
                    "in parallel across different learned projections of Q, K, and V. The h outputs are "
                    "concatenated and projected back to the model dimension. This lets the model attend "
                    "to information from multiple representation subspaces simultaneously."
                ),
                "model": MODEL,
                "stop_reason": "end_turn",
                "usage": 490,
                "rating": 5,
                "feedback_text": "Clear and concise — loved the formula and the intuition together.",
            },
        ],
    },
    {
        "user": "seed_alice",
        "name": "Multi-Head vs Single-Head Attention",
        "messages": [
            {
                "role": "user",
                "content": "What is the advantage of multi-head attention over single-head attention?",
            },
            {
                "role": "assistant",
                "content": (
                    "Single-head attention produces one set of attention weights, forcing the model to "
                    "summarise all relevant relationships into a single weighted average. Multi-head "
                    "attention instead runs the attention function h times in parallel, each with its "
                    "own learned linear projections for Q, K, and V.\n\n"
                    "The benefit is that different heads can specialise: one head might learn syntactic "
                    "dependencies, another co-reference links, another positional proximity. The paper "
                    "uses h=8 heads with d_k = d_model/h = 64, keeping total compute comparable to a "
                    "single full-dimension head."
                ),
                "model": MODEL,
                "stop_reason": "end_turn",
                "usage": 380,
                "rating": 4,
                "feedback_text": None,
            },
            {
                "role": "user",
                "content": "Does increasing the number of heads always help?",
            },
            {
                "role": "assistant",
                "content": (
                    "Not necessarily. More heads means each head operates in a lower-dimensional "
                    "subspace (since d_k = d_model / h), which can limit expressive capacity per head. "
                    "Empirically, there's a sweet spot — in the original paper, h=8 outperformed both "
                    "h=1 and much larger values on translation tasks. Later work (e.g., pruning studies) "
                    "found that many heads in trained models can be removed with minimal performance "
                    "loss, suggesting not all heads learn distinct patterns."
                ),
                "model": MODEL,
                "stop_reason": "end_turn",
                "usage": 295,
                "rating": 5,
                "feedback_text": "Exactly what I wanted — practical follow-up answered well.",
            },
        ],
    },
    {
        "user": "seed_bob",
        "name": "Word2Vec Skip-Gram Model",
        "messages": [
            {
                "role": "user",
                "content": "How does the Skip-Gram model in Word2Vec learn word embeddings?",
            },
            {
                "role": "assistant",
                "content": (
                    "Skip-Gram trains a shallow neural network to predict the surrounding context words "
                    "given a target word. For a word w_t at position t with window size c, the objective "
                    "is to maximise the log probability of context words w_{t-c} … w_{t+c}.\n\n"
                    "The network has a single hidden layer: the target word is one-hot encoded, "
                    "multiplied by an embedding matrix W (the word vectors being learned), then the "
                    "context is predicted via a second matrix W'. After training, the rows of W are "
                    "the word embeddings.\n\n"
                    "In practice, hierarchical softmax or negative sampling replaces the expensive "
                    "full softmax over the entire vocabulary."
                ),
                "model": MODEL,
                "stop_reason": "end_turn",
                "usage": 410,
                "rating": 4,
                "feedback_text": None,
            },
        ],
    },
    {
        "user": "seed_bob",
        "name": "BERT Masked Language Modeling",
        "messages": [
            {
                "role": "user",
                "content": "What is masked language modeling and how does BERT use it?",
            },
            {
                "role": "assistant",
                "content": (
                    "Masked Language Modeling (MLM) is a pre-training objective where a fraction of "
                    "input tokens are replaced with a [MASK] token and the model must predict the "
                    "original token from context.\n\n"
                    "BERT masks 15% of tokens. Of those, 80% become [MASK], 10% are replaced with a "
                    "random token, and 10% are left unchanged. The random-token and unchanged cases "
                    "prevent the model from learning to only attend to [MASK] positions.\n\n"
                    "Because BERT uses a bidirectional Transformer encoder, each masked position is "
                    "predicted using context from both left and right — unlike GPT-style left-to-right "
                    "language models. This gives BERT richer contextual representations."
                ),
                "model": MODEL,
                "stop_reason": "end_turn",
                "usage": 430,
                "rating": 5,
                "feedback_text": "Perfect breakdown of the 80/10/10 split, I always forget the details.",
            },
        ],
    },
    {
        "user": "seed_bob",
        "name": "BERT Next Sentence Prediction",
        "messages": [
            {
                "role": "user",
                "content": "Does BERT use any training objective beyond masked language modeling?",
            },
            {
                "role": "assistant",
                "content": (
                    "Yes — BERT also uses Next Sentence Prediction (NSP). During pre-training, "
                    "the model receives pairs of sentences and must predict whether the second sentence "
                    "follows the first in the original document (label: IsNext) or is a random sentence "
                    "(label: NotNext). 50% of pairs are genuine continuations; 50% are random.\n\n"
                    "NSP is intended to help tasks requiring sentence-pair reasoning like NLI and "
                    "question answering. Later work (RoBERTa, ALBERT) questioned its value, finding "
                    "that removing NSP often improves downstream task performance — suggesting MLM "
                    "alone provides sufficient training signal."
                ),
                "model": MODEL,
                "stop_reason": "end_turn",
                "usage": 350,
                "rating": 3,
                "feedback_text": "Good but I wish it had more detail on why NSP was later found unhelpful.",
            },
        ],
    },
    {
        "user": "seed_carol",
        "name": "VAE Latent Space Intuition",
        "messages": [
            {
                "role": "user",
                "content": "Can you give me an intuition for what the latent space in a VAE represents?",
            },
            {
                "role": "assistant",
                "content": (
                    "The latent space in a VAE is a continuous, structured representation of your data. "
                    "Each point in it corresponds to a plausible data sample — nearby points decode to "
                    "similar outputs.\n\n"
                    "The key difference from a plain autoencoder is the regularisation: the encoder "
                    "outputs parameters (μ, σ) of a Gaussian rather than a single point. A sample z "
                    "is drawn from N(μ, σ²) and passed to the decoder. The KL divergence term in the "
                    "loss pushes the per-datapoint posteriors toward a standard normal N(0,1), "
                    "preventing the latent space from collapsing into disconnected clusters.\n\n"
                    "The result is a smooth, interpolable space: you can walk between two points and "
                    "decode meaningful intermediate samples."
                ),
                "model": MODEL,
                "stop_reason": "end_turn",
                "usage": 460,
                "rating": 5,
                "feedback_text": "The smoothness explanation finally made it click for me.",
            },
        ],
    },
    {
        "user": "seed_carol",
        "name": "VAE Reparameterization Trick",
        "messages": [
            {
                "role": "user",
                "content": "Why does the VAE need a reparameterization trick? Can't you just sample z directly?",
            },
            {
                "role": "assistant",
                "content": (
                    "Sampling z ~ N(μ, σ²) directly breaks the computation graph. The sampling "
                    "operation itself has no gradient with respect to μ and σ, so backpropagation "
                    "cannot flow through it to update the encoder.\n\n"
                    "The reparameterization trick sidesteps this by rewriting the sample as:\n\n"
                    "    z = μ + σ · ε,  where ε ~ N(0, 1)\n\n"
                    "Now z is a deterministic function of μ and σ (plus ε, which is sampled from a "
                    "fixed distribution independent of the parameters). Gradients flow through μ and "
                    "σ normally, enabling end-to-end training via standard backprop."
                ),
                "model": MODEL,
                "stop_reason": "end_turn",
                "usage": 330,
                "rating": 5,
                "feedback_text": "Best explanation of the reparameterization trick I have read.",
            },
        ],
    },
    {
        "user": "seed_alice",
        "name": "Vision Transformer Patch Embeddings",
        "messages": [
            {
                "role": "user",
                "content": "How does the Vision Transformer (ViT) handle image input differently from a CNN?",
            },
            {
                "role": "assistant",
                "content": (
                    "CNNs process images through convolutional filters that slide across the spatial "
                    "dimensions, building up hierarchical local features.\n\n"
                    "ViT instead splits the image into a grid of fixed-size patches (e.g., 16×16 "
                    "pixels), flattens each patch into a vector, and linearly projects it to the model "
                    "dimension. These patch embeddings, plus a learned positional encoding and a "
                    "prepended [CLS] token, form the input sequence to a standard Transformer encoder.\n\n"
                    "There is no convolution at all. The self-attention mechanism handles all spatial "
                    "mixing, but does so globally from the first layer — unlike CNNs whose receptive "
                    "field grows gradually through depth."
                ),
                "model": MODEL,
                "stop_reason": "end_turn",
                "usage": 400,
                "rating": 4,
                "feedback_text": None,
            },
        ],
    },
    {
        "user": "seed_alice",
        "name": "ViT Data Requirements",
        "messages": [
            {
                "role": "user",
                "content": "I heard ViT needs a lot of data to train well. Is that true?",
            },
            {
                "role": "assistant",
                "content": (
                    "Yes, that is a notable finding from the paper. When trained on mid-size datasets "
                    "like ImageNet (~1.3M images), ViT underperforms comparably-sized ResNets. "
                    "CNNs have inductive biases baked in — translation equivariance and locality — "
                    "that help them generalise from less data.\n\n"
                    "ViT lacks these priors. To compensate, it needs large-scale pre-training: the "
                    "paper used JFT-300M (300M images), at which point ViT-Large/16 matches or exceeds "
                    "ResNets. The key insight is that given enough data, learned attention patterns can "
                    "substitute for hand-designed inductive biases."
                ),
                "model": MODEL,
                "stop_reason": "end_turn",
                "usage": 360,
                "rating": 2,
                "feedback_text": "Correct but feels like it's just restating the abstract. Wanted more depth on why the inductive biases matter less at scale.",
            },
        ],
    },
    {
        "user": "seed_bob",
        "name": "ResNet Skip Connections",
        "messages": [
            {
                "role": "user",
                "content": "What problem do skip connections solve in ResNet?",
            },
            {
                "role": "assistant",
                "content": (
                    "Very deep networks (>20 layers) suffered from a degradation problem: adding more "
                    "layers made training accuracy worse, even on the training set. This is not "
                    "overfitting — it is an optimisation problem.\n\n"
                    "Skip connections let layers learn a residual mapping F(x) = H(x) - x rather than "
                    "learning H(x) directly. If the optimal transformation is close to the identity, "
                    "it is easier to push F toward zero than to fit H(x) ≈ x from scratch.\n\n"
                    "In practice this means gradients have a direct path back through the network, "
                    "alleviating vanishing gradients and allowing networks of 100+ layers to train "
                    "reliably."
                ),
                "model": MODEL,
                "stop_reason": "end_turn",
                "usage": 390,
                "rating": 5,
                "feedback_text": "Nailed it — residual formulation explanation is exactly right.",
            },
        ],
    },
    {
        "user": "seed_carol",
        "name": "CLIP Contrastive Training",
        "messages": [
            {
                "role": "user",
                "content": "How does CLIP train its image and text encoders jointly?",
            },
            {
                "role": "assistant",
                "content": (
                    "CLIP uses contrastive learning on a dataset of (image, text) pairs scraped from "
                    "the web. For a batch of N pairs, it encodes all N images and N texts into "
                    "normalised embedding vectors.\n\n"
                    "It then computes an N×N matrix of cosine similarities between every image and "
                    "every text. The training objective is a symmetric cross-entropy loss: for each "
                    "image, maximise similarity to its matching text and minimise similarity to the "
                    "N-1 non-matching texts, and vice versa.\n\n"
                    "This pushes matching pairs together and non-matching pairs apart in the shared "
                    "embedding space. At inference, zero-shot classification works by encoding class "
                    "names as text and picking the class whose embedding is closest to the image."
                ),
                "model": MODEL,
                "stop_reason": "end_turn",
                "usage": 420,
                "rating": 4,
                "feedback_text": "Good overview, would have liked an example of the zero-shot prompt format.",
            },
        ],
    },
    {
        "user": "seed_carol",
        "name": "Transformer Positional Encoding",
        "messages": [
            {
                "role": "user",
                "content": "Why does the Transformer need positional encodings if it already uses attention?",
            },
            {
                "role": "assistant",
                "content": (
                    "Attention is permutation-equivariant: if you shuffle the input tokens, the "
                    "outputs shuffle in exactly the same way. There is no mechanism within the "
                    "attention operation itself that tells the model which position a token occupies.\n\n"
                    "Positional encodings inject position information by adding a position-dependent "
                    "vector to each token embedding before the first layer. The original paper uses "
                    "fixed sinusoidal encodings:\n\n"
                    "    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))\n"
                    "    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))\n\n"
                    "Different frequencies mean the model can attend to relative positions via linear "
                    "combinations of these values. Learned positional embeddings are an alternative "
                    "used by BERT and many later models."
                ),
                "model": MODEL,
                "stop_reason": "end_turn",
                "usage": 445,
                "rating": 5,
                "feedback_text": None,
            },
        ],
    },
    {
        "user": "seed_alice",
        "name": "KL Divergence in VAE Loss",
        "messages": [
            {
                "role": "user",
                "content": "What role does KL divergence play in the VAE loss function?",
            },
            {
                "role": "assistant",
                "content": (
                    "The VAE loss has two terms:\n\n"
                    "1. Reconstruction loss: how well the decoder reproduces the input from z.\n"
                    "2. KL divergence: KL(q(z|x) || p(z)), where p(z) = N(0,1).\n\n"
                    "The KL term acts as a regulariser on the latent space. Without it, the encoder "
                    "could encode each input into a tiny, isolated cluster — perfect reconstruction "
                    "but a useless, fragmented latent space with no generative structure.\n\n"
                    "The KL term penalises deviations from the prior N(0,1), pushing all posteriors "
                    "toward overlapping, smooth regions. This forces the decoder to produce "
                    "meaningful outputs across the whole latent space, enabling generation by sampling "
                    "from p(z) directly."
                ),
                "model": MODEL,
                "stop_reason": "end_turn",
                "usage": 410,
                "rating": 1,
                "feedback_text": "Too abstract. I asked about the loss function and got philosophy. Needs equations.",
            },
        ],
    },
    {
        "user": "seed_bob",
        "name": "BERT vs GPT Architecture Differences",
        "messages": [
            {
                "role": "user",
                "content": "What is the core architectural difference between BERT and GPT?",
            },
            {
                "role": "assistant",
                "content": (
                    "Both use the Transformer architecture but differ in which part they use and how "
                    "they apply attention.\n\n"
                    "GPT uses the Transformer decoder with causal (left-to-right) self-attention: "
                    "each token can only attend to previous tokens. This makes it naturally suited "
                    "for autoregressive generation.\n\n"
                    "BERT uses the Transformer encoder with full (bidirectional) self-attention: "
                    "every token attends to every other token. This is richer for representation "
                    "learning but cannot generate text autoregressively.\n\n"
                    "The downstream implication is that BERT excels at classification and extraction "
                    "tasks (fine-tuned with a head on [CLS]), while GPT is the foundation for "
                    "generation-based tasks."
                ),
                "model": MODEL,
                "stop_reason": "end_turn",
                "usage": 395,
                "rating": 4,
                "feedback_text": None,
            },
        ],
    },
    {
        "user": "seed_carol",
        "name": "ResNet vs Plain Network Depth",
        "messages": [
            {
                "role": "user",
                "content": "How does ResNet compare to a plain deep network of the same depth?",
            },
            {
                "role": "assistant",
                "content": (
                    "The paper directly compares ResNets against 'plain' networks (same architecture "
                    "but without skip connections) at 18, 34, 50, and 152 layers.\n\n"
                    "Key findings:\n"
                    "- Plain 34-layer networks have higher training error than plain 18-layer ones — "
                    "the degradation problem.\n"
                    "- ResNet-34 has lower training error than ResNet-18, showing skip connections "
                    "allow depth to actually help.\n"
                    "- ResNet-152 achieves 3.57% top-5 error on ImageNet, winning ILSVRC 2015.\n\n"
                    "The gain is entirely attributable to the residual connections, not to any other "
                    "architectural change."
                ),
                "model": MODEL,
                "stop_reason": "end_turn",
                "usage": 370,
                "rating": 4,
                "feedback_text": "Good concrete numbers from the paper.",
            },
        ],
    },
]


def _download_pdf(arxiv_id: str) -> bytes:
    """Download a PDF from arxiv. Returns raw bytes."""
    url = f"https://arxiv.org/pdf/{arxiv_id}"
    req = urllib.request.Request(url, headers={"User-Agent": "AquiLLM-seed-script/1.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.read()


def _extract_text(pdf_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(pdf_bytes))
    text = "".join(page.extract_text() + "\n" for page in reader.pages)
    return text.replace("\0", "")


class Command(BaseCommand):
    help = "Seed a dev instance with example PDFs, conversations, and feedback"

    def add_arguments(self, parser):
        parser.add_argument(
            "--force",
            action="store_true",
            help="Delete existing seed data and recreate from scratch",
        )

    def handle(self, *args, **options):
        force = options["force"]

        # ── Guard: already seeded? ─────────────────────────────────────────────
        existing_seed_users = User.objects.filter(username__startswith=SEED_TAG)
        if existing_seed_users.exists():
            if not force:
                self.stdout.write(self.style.WARNING(
                    "Seed data already exists. Run with --force to delete and recreate."
                ))
                return
            self.stdout.write("--force: deleting existing seed data…")
            # PDFDocument.ingested_by uses on_delete=RESTRICT, so we must delete
            # all documents uploaded by seed users before deleting the users.
            PDFDocument.objects.filter(ingested_by__in=existing_seed_users).delete()
            # Remove seed collections
            for col_def in COLLECTION_DEFS:
                Collection.objects.filter(name=col_def["name"]).delete()
            # Deleting users cascades to their conversations and messages.
            existing_seed_users.delete()
            self.stdout.write("  Done.\n")

        # ── 1. Users ───────────────────────────────────────────────────────────
        self.stdout.write("Creating seed users…")
        users: dict[str, User] = {}
        for u in SEED_USERS:
            user = User.objects.create_user(
                username=u["username"],
                email=u["email"],
                password="seedpassword123",
                first_name=u["first_name"],
                last_name=u["last_name"],
            )
            users[u["username"]] = user
            self.stdout.write(f"  Created user: {u['username']}")

        # ── 2. Collections ─────────────────────────────────────────────────────
        self.stdout.write("\nCreating collections…")
        collections: dict[str, Collection] = {}
        for col_def in COLLECTION_DEFS:
            col = Collection.objects.create(name=col_def["name"])
            CollectionPermission.objects.create(
                user=users[col_def["owner"]], collection=col, permission="MANAGE"
            )
            for username in col_def.get("editors", []):
                CollectionPermission.objects.create(
                    user=users[username], collection=col, permission="EDIT"
                )
            for username in col_def.get("viewers", []):
                CollectionPermission.objects.create(
                    user=users[username], collection=col, permission="VIEW"
                )
            collections[col_def["name"]] = col
            self.stdout.write(f"  Created collection: {col_def['name']}")

        # ── 3. PDFs ────────────────────────────────────────────────────────────
        self.stdout.write("\nDownloading and ingesting arxiv papers…")
        for paper in ARXIV_PAPERS:
            self.stdout.write(f"  Fetching {paper['arxiv_id']}: {paper['title'][:60]}…")
            try:
                pdf_bytes = _download_pdf(paper["arxiv_id"])
                full_text = _extract_text(pdf_bytes)
            except Exception as exc:
                self.stdout.write(self.style.ERROR(f"    Failed to download: {exc}"))
                continue

            doc = PDFDocument(
                title=paper["title"],
                collection=collections[paper["collection"]],
                ingested_by=users[paper["uploader"]],
                full_text=full_text,
            )
            # Upload PDF bytes to MinIO via Django file storage, then save the model.
            # full_text is already set so PDFDocument.save() skips re-extraction.
            doc.pdf_file.save(
                f"{paper['arxiv_id']}.pdf",
                ContentFile(pdf_bytes),
                save=False,
            )
            doc.save()
            self.stdout.write(self.style.SUCCESS(f"    Ingested ({len(pdf_bytes)//1024} KB, {len(full_text.split())} words)"))

        # ── 4. Conversations + messages ────────────────────────────────────────
        self.stdout.write("\nCreating conversations and messages…")
        now = timezone.now()

        for i, convo_def in enumerate(CONVERSATIONS):
            # Spread conversations over the past 30 days
            created_offset = timedelta(days=random.randint(0, 30), hours=random.randint(0, 23))
            created_at = now - created_offset

            convo = WSConversation.objects.create(
                owner=users[convo_def["user"]],
                name=convo_def["name"],
                created_at=created_at,
                updated_at=created_at,
            )

            for seq, msg_def in enumerate(convo_def["messages"]):
                msg_created_at = created_at + timedelta(minutes=seq * 2)
                feedback_at = None
                if msg_def.get("rating") is not None:
                    feedback_at = msg_created_at + timedelta(minutes=random.randint(1, 10))

                Message.objects.create(
                    conversation=convo,
                    role=msg_def["role"],
                    content=msg_def["content"],
                    sequence_number=seq,
                    model=msg_def.get("model"),
                    stop_reason=msg_def.get("stop_reason"),
                    usage=msg_def.get("usage", 0),
                    rating=msg_def.get("rating"),
                    feedback_text=msg_def.get("feedback_text"),
                    feedback_submitted_at=feedback_at,
                    created_at=msg_created_at,
                )

            self.stdout.write(f"  [{i+1:02d}/{len(CONVERSATIONS)}] {convo_def['name']}")

        # ── Summary ────────────────────────────────────────────────────────────
        self.stdout.write(self.style.SUCCESS("\n✓ Seed data created"))
        self.stdout.write(f"  Users       : {len(SEED_USERS)}")
        self.stdout.write(f"  Collections : {len(COLLECTION_DEFS)}")
        self.stdout.write(f"  Papers      : {len(ARXIV_PAPERS)}")
        self.stdout.write(f"  Conversations: {len(CONVERSATIONS)}")
        total_messages = sum(len(c["messages"]) for c in CONVERSATIONS)
        rated = sum(
            1 for c in CONVERSATIONS for m in c["messages"] if m.get("rating") is not None
        )
        feedback_text_count = sum(
            1 for c in CONVERSATIONS for m in c["messages"] if m.get("feedback_text")
        )
        self.stdout.write(f"  Messages    : {total_messages} ({rated} rated, {feedback_text_count} with text feedback)")
        self.stdout.write("\nRun the server and log in as seed_alice / seed_bob / seed_carol (password: seedpassword123)")
        self.stdout.write("Note: PDF chunking runs async via Celery — documents may take a moment to become searchable.")
