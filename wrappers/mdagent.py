import re

from wrappers.agent import AgentMetaWrapper


class MDAgentWrapper(AgentMetaWrapper):
    """Multi-agent medical reasoning wrapper over any chat-capable VLM backbone."""

    def __init__(self, backbone, args):
        super().__init__(backbone=backbone, args=args)

        self.name = f"MDAgent({getattr(backbone, 'name', args.model)})"

    def infer_vision_language(self, image, qs, image_size=None):
        mode = self._resolve_mode(qs)

        if mode == "basic":
            answer = self._run_basic(image, qs, image_size=image_size)
        elif mode == "advanced":
            answer = self._run_advanced(image, qs, image_size=image_size)
        else:
            answer = self._run_intermediate(image, qs, image_size=image_size)

        self.last_trace["mode"] = mode
        self.last_trace["final_answer"] = answer
        return answer

    def _resolve_mode(self, qs):
        requested_mode = getattr(self.args, "mdagent_mode", "adaptive")
        if requested_mode != "adaptive":
            return requested_mode

        question = qs.lower().strip()
        token_count = len(question.split())
        multichoice_markers = [
            "answer choices:",
            "answer with the single letter",
            "(a)",
            "(b)",
            "(c)",
        ]
        if any(marker in question for marker in multichoice_markers):
            return "basic"

        yes_no_patterns = [
            r"^answer the following question about the image with yes or no",
            r"\byes or no\b",
            r"^(is|are|does|do|can|could|was|were|has|have)\b",
        ]
        if any(re.search(pattern, question) for pattern in yes_no_patterns):
            return "basic"

        advanced_markers = [
            "why",
            "explain",
            "describe",
            "evidence",
            "differential",
            "most likely diagnosis",
            "best next",
            "how many",
            "comparison",
        ]
        if token_count >= 18 or any(marker in question for marker in advanced_markers):
            return "advanced"

        return "intermediate"

    def _query_backbone(self, image, qs, role, task, evidence=None, image_size=None):
        prompt_parts = [
            f"You are acting as the {role} in a medical multi-agent team.",
            "Use only the provided image and question.",
            "If the image is uncertain or insufficient, say so briefly and still provide the best supported answer.",
            f"Task: {task}",
            f"Question: {qs}",
        ]
        if evidence:
            prompt_parts.append(f"Prior team notes:\n{evidence}")

        prompt = "\n\n".join(prompt_parts).strip()

        return self.backbone.infer_vision_language(image, prompt, image_size=image_size)

    def _run_basic(self, image, qs, image_size=None):
        answer = self._query_backbone(
            image=image,
            qs=qs,
            role="primary clinician",
            task="Answer the question directly and concisely. Return only the final answer.",
            image_size=image_size,
        )
        self.last_trace = {
            "steps": [
                {
                    "agent": "primary_clinician",
                    "output": answer,
                }
            ]
        }
        return answer

    def _run_intermediate(self, image, qs, image_size=None):
        findings = self._query_backbone(
            image=image,
            qs=qs,
            role="visual examiner",
            task="List the most relevant visual findings for answering the question. Keep it short.",
            image_size=image_size,
        )
        proposal = self._query_backbone(
            image=image,
            qs=qs,
            role="medical specialist",
            task="Use the image and the prior team notes to propose the best answer with a short rationale.",
            evidence=findings,
            image_size=image_size,
        )
        critique = self._query_backbone(
            image=image,
            qs=qs,
            role="skeptical reviewer",
            task="Check the proposed answer for mistakes, unsupported claims, or better alternatives. Be concise.",
            evidence=f"[visual examiner]\n{findings}\n\n[medical specialist]\n{proposal}",
            image_size=image_size,
        )
        answer = self._query_backbone(
            image=image,
            qs=qs,
            role="moderator",
            task="Synthesize the team discussion and return only the final answer in a concise form.",
            evidence=(
                f"[visual examiner]\n{findings}\n\n"
                f"[medical specialist]\n{proposal}\n\n"
                f"[skeptical reviewer]\n{critique}"
            ),
            image_size=image_size,
        )

        self.last_trace = {
            "steps": [
                {"agent": "visual_examiner", "output": findings},
                {"agent": "medical_specialist", "output": proposal},
                {"agent": "skeptical_reviewer", "output": critique},
                {"agent": "moderator", "output": answer},
            ]
        }
        return answer

    def _run_advanced(self, image, qs, image_size=None):
        plan = self._query_backbone(
            image=image,
            qs=qs,
            role="triage lead",
            task="Break the problem into 2-4 short subproblems needed to answer the question.",
            image_size=image_size,
        )
        findings = self._query_backbone(
            image=image,
            qs=qs,
            role="visual examiner",
            task="Summarize the clinically relevant visual findings that matter for the question.",
            evidence=plan,
            image_size=image_size,
        )
        differential = self._query_backbone(
            image=image,
            qs=qs,
            role="differential diagnostician",
            task="Propose the best answer and key alternatives, grounded in the image and notes.",
            evidence=f"[triage plan]\n{plan}\n\n[visual findings]\n{findings}",
            image_size=image_size,
        )
        verifier = self._query_backbone(
            image=image,
            qs=qs,
            role="quality reviewer",
            task="Identify weak points in the proposed answer and state whether it should be revised.",
            evidence=(
                f"[triage plan]\n{plan}\n\n"
                f"[visual findings]\n{findings}\n\n"
                f"[differential diagnostician]\n{differential}"
            ),
            image_size=image_size,
        )
        answer = self._query_backbone(
            image=image,
            qs=qs,
            role="chief moderator",
            task="Return only the final answer. Keep it concise and directly answer the question.",
            evidence=(
                f"[triage plan]\n{plan}\n\n"
                f"[visual findings]\n{findings}\n\n"
                f"[differential diagnostician]\n{differential}\n\n"
                f"[quality reviewer]\n{verifier}"
            ),
            image_size=image_size,
        )

        self.last_trace = {
            "steps": [
                {"agent": "triage_lead", "output": plan},
                {"agent": "visual_examiner", "output": findings},
                {"agent": "differential_diagnostician", "output": differential},
                {"agent": "quality_reviewer", "output": verifier},
                {"agent": "chief_moderator", "output": answer},
            ]
        }
        return answer
