"""Correction handler — applies parsed correction intents to PCR state."""

from app.core.pcr_state_manager import PCRStateManager
from app.schemas.correction import CorrectionIntent
from app.schemas.pcr import PCRStateEnvelope
from app.utils.logging import logger


class CorrectionHandler:
    """Apply parsed corrections to PCR state."""

    def apply(
        self,
        state_manager: PCRStateManager,
        intents: list[CorrectionIntent],
    ) -> tuple[PCRStateEnvelope, list[dict]]:
        """Apply correction intents and return updated state + rejected intents.

        Returns:
            tuple of (updated PCRStateEnvelope, list of rejected intent dicts)
        """
        rejected = []
        state = state_manager.get_state()

        for intent in intents:
            # Validate field exists in PCR
            if not hasattr(state.pcr, intent.field):
                rejected.append({
                    "intent": intent.model_dump(),
                    "reason": f"Unknown field: {intent.field}",
                })
                continue

            # Skip low-confidence corrections
            if intent.confidence < 0.5:
                rejected.append({
                    "intent": intent.model_dump(),
                    "reason": f"Confidence too low: {intent.confidence}",
                })
                continue

            try:
                state = state_manager.apply_correction(
                    field_name=intent.field,
                    new_value=intent.value,
                    action=intent.action,
                )
                logger.info(
                    f"Applied correction: {intent.field} = {intent.value} "
                    f"(action={intent.action})"
                )
            except Exception as e:
                rejected.append({
                    "intent": intent.model_dump(),
                    "reason": str(e),
                })
                logger.error(f"Failed to apply correction: {e}")

        return state, rejected
