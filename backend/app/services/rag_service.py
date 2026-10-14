"""End-to-end RAG orchestration."""

import logging
from collections.abc import Sequence

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.core_logging import debug_logger
from app.core.exceptions import GroundingError
from app.rag.models import RetrievalResult
from app.schemas.chat import ChatMessage, ChatResponse
from app.schemas.policy import PolicyCitation
from app.services.audit_service import AuditService
from app.services.conversation_service import ConversationService
from app.services.generation_service import GenerationService
from app.services.prompt_service import build_generation_prompt
from app.services.query_rewriter import QueryRewriterService
from app.services.retrieval_service import RetrievalService
from app.services.validation_service import ValidationService

logger = logging.getLogger(__name__)
dev_log = debug_logger(__name__)


# The exact fallback string the LLM is instructed to return when evidence
# is truly absent. If the LLM returns this, we treat it as a VALID response
# (not a grounding failure) and return it directly to the user.
FALLBACK_MARKER = "I couldn't find this information in the retrieved policy sections"


# ==============================================================
# commented the below block to remove the hardcoded greetings answers
# ==============================================================

# # ---------------------------------------------------------------------------
# # Content-free query detection
# # ---------------------------------------------------------------------------
# # A "content token" is a lowercase alphanumeric word of 3+ characters that
# # is NOT in the function-word set below. Queries that have zero content
# # tokens (like "hi", "thanks", "bye") cannot be answered from policy
# # evidence — they should short-circuit to a friendly reply without
# # burning an LLM call or triggering a grounding failure.
# #
# # The function-word list is deliberately conservative. It only contains
# # words that cannot carry policy meaning. When in doubt, a word is NOT in
# # the list, so its presence marks the query as contentful and lets it
# # through to the RAG pipeline. A false positive (greeting sent to RAG) is
# # harmless; a false negative (real question short-circuited) is not.
# _CONTENT_TOKEN_RE = re.compile(r"[a-z0-9]{3,}")
# _FUNCTION_WORDS: frozenset[str] = frozenset(
#     {
#         # Common English function words
#         "the",
#         "and",
#         "are",
#         "but",
#         "for",
#         "how",
#         "its",
#         "not",
#         "you",
#         "your",
#         "with",
#         "what",
#         "when",
#         "where",
#         "which",
#         "who",
#         "why",
#         "will",
#         "would",
#         "could",
#         "should",
#         "there",
#         "their",
#         "them",
#         "they",
#         "this",
#         "that",
#         "these",
#         "those",
#         "have",
#         "has",
#         "had",
#         "was",
#         "were",
#         "been",
#         "being",
#         "into",
#         "from",
#         "about",
#         "over",
#         "under",
#         "just",
#         "some",
#         "any",
#         "all",
#         "one",
#         "two",
#         "get",
#         "got",
#         "let",
#         "put",
#         "say",
#         "see",
#         "way",
#         "use",
#         "ask",
#         # Greetings
#         "hey",
#         "heya",
#         "hi",
#         "hii",
#         "hiya",
#         "hello",
#         "helo",
#         "yo",
#         "sup",
#         "greetings",
#         "howdy",
#         "gday",
#         # Thanks
#         "thanks",
#         "thank",
#         "thankyou",
#         "thx",
#         "ty",
#         "ta",
#         "cheers",
#         # Goodbyes
#         "bye",
#         "byee",
#         "goodbye",
#         "cya",
#         "seeya",
#         "farewell",
#         # Acknowledgements
#         "ok",
#         "okay",
#         "yep",
#         "yeah",
#         "yup",
#         "nope",
#         "nah",
#         "cool",
#         "nice",
#         "sure",
#         "right",
#         "fine",
#     }
# )


# def _has_content_tokens(query: str) -> bool:
#     """Return True if the query contains at least one meaningful word."""
#     tokens = _CONTENT_TOKEN_RE.findall((query or "").lower())
#     return any(t not in _FUNCTION_WORDS for t in tokens)


class RAGService:
    """Execute the complete grounded policy-question workflow."""

    def __init__(
        self,
        retrieval_service: RetrievalService,
        generation_service: GenerationService,
        validation_service: ValidationService,
        audit_service: AuditService,
        query_rewriter: QueryRewriterService | None = None,
    ) -> None:
        self.settings = get_settings()
        self.retrieval_service = retrieval_service
        self.generation_service = generation_service
        self.validation_service = validation_service
        self.audit_service = audit_service
        self.query_rewriter = query_rewriter

    async def answer(
        self,
        session: AsyncSession,
        *,
        query: str,
        conversation: Sequence[ChatMessage],
        user_id: int,
        request_id: str,
        conversation_id: int | None = None,
    ) -> ChatResponse:
        """Generate a safe, cited answer."""

        # ==============================================================
        # commented the below block to remove the hardcoded greetings answers
        # ==============================================================

        # # ── STEP -1: Content-free query guard ────────────────────────
        # # Greetings, thanks, goodbyes and similar have no content tokens
        # # and cannot be answered from policy evidence. Short-circuit them
        # # here so they don't burn tokens, produce no citations, and
        # # trigger a GroundingError further down the pipeline.
        # if not _has_content_tokens(query):
        #     answer = (
        #         "Hi! I'm here to help with La Trobe University policies, "
        #         "procedures, and staff documentation. What would you like "
        #         'to know? For example: "What is the special consideration '
        #         'process?" or "How do I apply for a short extension?"'
        #     )

        #     await self.audit_service.record(
        #         session,
        #         request_id=request_id,
        #         event_type="RAG_TRIVIAL_QUERY",
        #         user_id=user_id,
        #         query=query,
        #         retrieved_chunk_count=0,
        #         retrieval_score=None,
        #         grounding_score=1.0,
        #         success=True,
        #     )

        #     saved_conversation_id = await self._persist_exchange(
        #         session,
        #         conversation_id=conversation_id,
        #         user_id=user_id,
        #         user_content=query,
        #         assistant_content=answer,
        #         assistant_citations=[],
        #         request_id=request_id,
        #     )

        #     return ChatResponse(
        #         answer=answer,
        #         citations=[],
        #         grounded=False,
        #         confidence=1.0,
        #         escalation_required=False,
        #         disclaimer=self.validation_service.disclaimer(),
        #         request_id=request_id,
        #         conversation_id=saved_conversation_id,
        #     )

        try:
            # ── STEP 0: Query rewriting for conversational follow-ups ─
            search_query = query
            if self.query_rewriter is not None and conversation:
                search_query = await self.query_rewriter.rewrite(query, conversation)

            dev_log.warning(
                "SEARCH QUERY | original=%r | effective=%r",
                query,
                search_query,
            )

            # ── STEP 1: Retrieval ────────────────────────────────────
            retrieved = await self.retrieval_service.retrieve(session, search_query)

            dev_log.warning(
                "RETRIEVAL | chunks=%d | ids=%s",
                len(retrieved),
                [r.chunk_id for r in retrieved],
            )

            # ── STEP 2: No-evidence fallback ─────────────────────────
            if not retrieved:
                answer = (
                    "I couldn't find this information in the retrieved "
                    "policy sections. Please contact the relevant "
                    "La Trobe University administration for clarification."
                )

                await self.audit_service.record(
                    session,
                    request_id=request_id,
                    event_type="RAG_NO_EVIDENCE",
                    user_id=user_id,
                    query=query,
                    retrieved_chunk_count=0,
                    retrieval_score=None,
                    grounding_score=0.0,
                    success=True,
                )

                saved_conversation_id = await self._persist_exchange(
                    session,
                    conversation_id=conversation_id,
                    user_id=user_id,
                    user_content=query,
                    assistant_content=answer,
                    assistant_citations=[],
                    request_id=request_id,
                )

                return ChatResponse(
                    answer=answer,
                    citations=[],
                    grounded=False,
                    confidence=0.0,
                    escalation_required=True,
                    disclaimer=self.validation_service.disclaimer(),
                    request_id=request_id,
                    conversation_id=saved_conversation_id,
                )

            # ── STEP 3: Build prompt ([C1]..[C8] assigned here) ──────
            prompt = build_generation_prompt(
                query=query,
                retrieved=retrieved,
                conversation=[
                    {"role": m.role, "content": m.content} for m in conversation
                ],
            )

            dev_log.warning(
                "PROMPT | length=%d | head=%r",
                len(prompt),
                prompt[:400],
            )

            # ── STEP 4: Generation ───────────────────────────────────
            generated = await self.generation_service.generate(prompt)

            dev_log.warning("LLM RAW | %r", generated)

            # ── STEP 5: Citation validation ──────────────────────────
            valid, confidence, citation_ids = self.validation_service.validate(
                generated,
                retrieved,
            )

            dev_log.warning(
                "VALIDATION | valid=%s | confidence=%s | citations=%s",
                valid,
                confidence,
                citation_ids,
            )

            # ── STEP 6a: Fallback path — treat as success ────────────
            if not valid and FALLBACK_MARKER in generated:
                dev_log.warning(
                    "FALLBACK PATH | returning insufficient-evidence message"
                )

                await self.audit_service.record(
                    session,
                    request_id=request_id,
                    event_type="RAG_NO_EVIDENCE",
                    user_id=user_id,
                    query=query,
                    retrieved_chunk_count=len(retrieved),
                    retrieval_score=retrieved[0].fused_score,
                    grounding_score=0.0,
                    success=True,
                )

                saved_conversation_id = await self._persist_exchange(
                    session,
                    conversation_id=conversation_id,
                    user_id=user_id,
                    user_content=query,
                    assistant_content=generated,
                    assistant_citations=[],
                    request_id=request_id,
                )

                return ChatResponse(
                    answer=generated,
                    citations=[],
                    grounded=False,
                    confidence=0.0,
                    escalation_required=True,
                    disclaimer=self.validation_service.disclaimer(),
                    request_id=request_id,
                    conversation_id=saved_conversation_id,
                )

            # ── STEP 6b: True grounding failure ──────────────────────
            if not valid:
                raise GroundingError(
                    "The generated response could not be sufficiently "
                    "grounded in the retrieved policy evidence."
                )

            # ── STEP 7: Build public citations ───────────────────────
            citations = self._build_citations(retrieved, citation_ids)

            # ── STEP 8: Escalation check ─────────────────────────────
            escalation_required = self.validation_service.needs_escalation(
                generated,
                confidence,
            )

            # ── STEP 9: Audit success ────────────────────────────────
            await self.audit_service.record(
                session,
                request_id=request_id,
                event_type="RAG_COMPLETED",
                user_id=user_id,
                query=query,
                retrieved_chunk_count=len(retrieved),
                retrieval_score=retrieved[0].fused_score,
                grounding_score=confidence,
                success=True,
            )

            # ── STEP 10: Persist the exchange ────────────────────────
            saved_conversation_id = await self._persist_exchange(
                session,
                conversation_id=conversation_id,
                user_id=user_id,
                user_content=query,
                assistant_content=generated,
                assistant_citations=citations,
                request_id=request_id,
            )

            # ── STEP 11: Return response ─────────────────────────────
            return ChatResponse(
                answer=generated,
                citations=citations,
                grounded=True,
                confidence=confidence,
                escalation_required=escalation_required,
                disclaimer=self.validation_service.disclaimer(),
                request_id=request_id,
                conversation_id=saved_conversation_id,
            )

        # ── GroundingError (our own validation failure) ──────────────
        except GroundingError:
            await self.audit_service.record(
                session,
                request_id=request_id,
                event_type="RAG_GROUNDING_FAILURE",
                user_id=user_id,
                query=query,
                retrieved_chunk_count=None,
                retrieval_score=None,
                grounding_score=0.0,
                success=False,
            )
            raise

        # ── Any other unexpected failure ─────────────────────────────
        except Exception:
            logger.exception("RAG pipeline failed. request_id=%s", request_id)

            await self.audit_service.record(
                session,
                request_id=request_id,
                event_type="RAG_FAILURE",
                user_id=user_id,
                query=query,
                retrieved_chunk_count=None,
                retrieval_score=None,
                grounding_score=None,
                success=False,
            )

            raise

    # ────────────────────────────────────────────────────────────────
    # Helpers
    # ────────────────────────────────────────────────────────────────

    async def _persist_exchange(
        self,
        session: AsyncSession,
        *,
        conversation_id: int | None,
        user_id: int,
        user_content: str,
        assistant_content: str,
        assistant_citations: list[PolicyCitation],
        request_id: str,
    ) -> int | None:
        """
        Save a user/assistant exchange to the user's conversation.

        - If `conversation_id` is None, a new conversation is created.
        - Returns the conversation id used, or None if persistence failed.
        - Non-fatal: exceptions are logged and swallowed so the answer
          is still returned to the user.
        """
        try:
            convo_service = ConversationService(session)

            saved_id = conversation_id
            if saved_id is None:
                new_conv = await convo_service.create(user_id)
                saved_id = new_conv.id

            await convo_service.append_exchange(
                conversation_id=saved_id,
                user_id=user_id,
                user_content=user_content,
                assistant_content=assistant_content,
                assistant_citations=assistant_citations,
            )
            await session.commit()
            return saved_id

        except Exception:
            await session.rollback()
            logger.exception(
                "Failed to persist conversation. request_id=%s", request_id
            )
            return conversation_id

    @staticmethod
    def _build_citations(
        retrieved: list[RetrievalResult],
        citation_ids: list[str],
    ) -> list[PolicyCitation]:
        """Convert internal citation IDs into public citation objects."""
        citations: list[PolicyCitation] = []

        for citation_id in citation_ids:
            index = int(citation_id[1:]) - 1

            if index < 0 or index >= len(retrieved):
                continue

            result = retrieved[index]

            citations.append(
                PolicyCitation(
                    citation_id=citation_id,
                    policy_id=result.policy_id,
                    title=result.title,
                    section=result.section,
                    url=result.url,
                )
            )

        return citations
