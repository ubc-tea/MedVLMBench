from copy import deepcopy
from collections import defaultdict
import re

from wrappers.agent import AgentMetaWrapper
from eval.utils import extract_choice_letter

MAX_RETRY = 1


class UCAgentWrapper(AgentMetaWrapper):
    """
    Hierarchical UC-Agent implemented under AgentMetaWrapper.
    Contains Level-1 / Level-2 / Level-3 diagnosis internally.
    """

    def __init__(self, backbone, args):
        super().__init__(backbone=backbone, args=args)

        self.name = f"UCAgent({getattr(backbone, 'name', args.model)})"

        # initialize system prompt (same logic as original Agent)
        self.system_prompt = (
            "Disclaimer: This task is a research-oriented, educational task. "
            "You must base your responses only on observable image features "
            "and logical reasoning principles."
        )

    def _single_expert_level1(self, question, image, temperature=0.7, expert_id=None):

        prompt = f"""
        [Core Identity] You are a professional and rigorous <MEDICAL FIELD> expert specializing in diagnostic imaging interpretation (<IMAGING MODALITIES>). Your core goal is to make precise, evidence-based diagnoses for the given question strictly based on the provided <IMAGING TYPE> image and medical case. 
        
        [Medical Case] {question}.

        [Reasoning Requirements] 
        1. Carefully examine the image and read the question.
        2. Identify the type of question:
            - "open" (free-form clinical question)
            - "yes/no" (binary question requiring yes or no)
            - "multi-choice" (multiple-choice question with labeled options such as A, B, C, D, etc.)
        3. Provide structured reasoning based only on visible image evidence and medical knowledge.
        4. Provide the final answer following the format constraints below. 
        
        [Answer Format Constraints]
        - If the question type is "open": provide a concise but complete medical answer (no format restriction).
        - If the question type is "yes/no": the answer MUST be exactly either "yes" or "no".
        - If the question type is "multi-choice": the answer MUST be a single uppercase letter (e.g., A, B, C, D).

        [Strict Output Format]
        #Question Type: <open / yes/no / multi-choice>
        #Reasoning: <1-2 concise sentences grounded in visible image evidence>
        #Answer: <final answer strictly following the format constraint>
        """

        retry_count = 0
        while retry_count < MAX_RETRY:
            response = self._query_backbone(image, prompt, temperature=temperature)
            self.last_trace[f"level1_expert{expert_id+1}"] = response

            extraction = self._extract_option(response)

            if extraction["validation_error"] is None:
                break  # ✅ success

            retry_count += 1

        if extraction is None or extraction["validation_error"] is not None:
            self.last_trace[f"error"] = {
                "question": question,
                "prompt": prompt,
                "response": response,
                "error": extraction["validation_error"] if extraction else "unknown",
            }

            raise ValueError(f"Agent formatting error: {self.last_trace}")

        return response, extraction

    # ==========================================================
    # Level 2
    # ==========================================================

    def _level2_diagnosis(self, question, image, level1_report):

        prompt = f"""
        [Core Identity] You are an authoritative senior <MEDICAL FIELD> expert, highly proficient in <IMAGING MODALITIES> interpretation and diagnostic reasoning. Your role is to critically verify the consensus diagnosis made by two prior <MEDICAL FIELD> experts, ensuring it is logically sound, evidence-based, and consistent with <IMAGING MODALITIES> image features.
        
        [Task Focus] 
        1. First check the input image and read the question. 
        2. Identify the type of question:
            - "open" (free-form clinical question)
            - "yes/no" (binary question requiring yes or no)
            - "multi-choice" (multiple-choice question with labeled options such as A, B, C, D, etc.)
        3. Evaluate whether the shared judgment aligns with the observed image findings and <IMAGING MODALITIES> criteria. 
        4. Identify any potential misinterpretation or overconfidence. 
        5. If their consensus is valid, reaffirm it; if not, provide your corrected final diagnosis. 
        
        [Current Case] {question}. 
        
        [Previous Reports] {level1_report}. 

        [Answer Format Constraints]
        - If the question type is "open": provide a concise but complete medical answer (no format restriction).
        - If the question type is "yes/no": the answer MUST be exactly either "yes" or "no".
        - If the question type is "multi-choice": the answer MUST be a single uppercase letter (e.g., A, B, C, D).
        
        [Strict Output Format] 
        #Question Type: <open / yes/no / multi-choice>
        #Reasoning: <Write 1-2 concise sentences covering the key evidence and your conclusion.> 
        #Answer: <final answer strictly following the format constraint>
        """

        retry_count = 0
        while retry_count < MAX_RETRY:
            response = self._query_backbone(image, prompt, temperature=0.5)
            self.last_trace[f"level2"] = response

            extraction = self._extract_option(response)

            if extraction["validation_error"] is None:
                break  # ✅ success

            retry_count += 1

        if extraction is None or extraction["validation_error"] is not None:
            self.last_trace[f"error"] = {
                "question": question,
                "prompt": prompt,
                "response": response,
                "error": extraction["validation_error"] if extraction else "unknown",
            }

            raise ValueError(f"Agent formatting error: {self.last_trace}")

        return response, extraction

    # ==========================================================
    # Level 3
    # ==========================================================

    def _level3_diagnosis(self, question, image, latest_report, level1_dict):
        critics = {}
        debate_history = {}

        # each option gets critic
        for option in level1_dict.keys():
            prompt = f"""
            [Core Identity] You are an expert Critical Analyst, functioning as a Hypothesis Auditor. First check the input image, and read the question. Your task is to provide a balanced, objective, and rigorous review of a proposed hypothesis based on the provided source evidence. Your goal is to assess the overall viability and logical soundness of the hypothesis, not to attack it. You are assigned to uncover potential risks in answer {option} in the medical case and the supportive statements of answer {option} in [Historical Reports]. You should raise the risk that "why this hypothesis may be wrong", and your report would be given to a leader to make a decision. 
            
            [Medical Case] {question}. 
            
            [Historical Reports] {latest_report}. 
            
            [Output Format] #Flaws: <Describe the main flaw or risk in 1-2 concise sentences.> Counter Evidence: <Cite supporting evidence from the case in 1-2 concise sentences.>.
            """

            critic_response = self._query_backbone(image, prompt, temperature=0.5)
            self.last_trace[f"level3_critic_{option}"] = critic_response
            critics[option] = critic_response
            debate_history[
                option
            ] = f"""
            # Your initial task: 
            {prompt}

            # Your initial support report:
            {critic_response}
            """

        # leader adjudication
        critic_text = "(LEVEL-3 Expert Panel Critics)\n" + "\n".join(
            [f"Critic Expert {i+1}: {critic}" for i, critic in enumerate(critics)]
        )

        leader_prompt = f"""
        [Core Identity] You are the Lead Adjudicator, responsible for chairing an expert critical analysis of conflicting hypotheses. You are impartial, perceptive, and skilled at uncovering the truth through precise inquiry. 
        
        [Task 1] First check the input image and read the question. You have just received the initial arguments on a medical case from the Critic Specialists. Your task is not to form your own opinion yet, but to act as a rigorous, impartial critic. You must critically analyze each review below, identify its single biggest weakness, logical flaw, or unsupported assumption, and formulate a targeted, challenging question for each specialist, the question should help you solve the case. 
        
        [Inquiry Methodology] Strictly follow these steps in your thinking: 1.Synthesize Critiques: Comprehensively read and understand the report submitted by each Hypothesis Auditor. 2.Identify Core Conflict: What is the central point of disagreement or the most critical identified risk among the competing audits? 3.Formulate Targeted Questions: Based on this core conflict, design a challenging question for each auditor that forces them to defend their critique. 
        
        [Output Format] Inquiries:@ To Expert <Expert No., e.g 1> who reviews Agent1: <The single, most pointed question for the Expert who reviews Answer, based on the risks they identified in their report.> @ To Expert <Expert No., e.g 2> who reviews Agent2: <The single, most pointed question for the Expert who reviews Answer>...(until each expert in [Critics on Assessments] is inquired, no other contents). 
        
        Rules:
        - Use EXACTLY the string "Agent1", "Agent2", etc. (NOT A/B).
        - Each inquiry MUST start with "@ To Expert".
        - Do NOT include any additional text before or after the inquiries.

        [Medical Case] {question}. 
        
        [Initial Independent Assessments] {latest_report}. 
        
        [Critics on Assessments] {critic_text}. 
        
        Now, begin your inquiry and output strictly according to the format and requirements:
        """

        retry_count = 0
        while retry_count < MAX_RETRY:
            leader_consulations = self._query_backbone(image, leader_prompt, temperature=0.1)
            self.last_trace[f"level3_leader"] = leader_consulations

            extraction = self._extract_inquiries(leader_consulations)

            if extraction["validation_error"] is None:
                break  # ✅ success

            retry_count += 1

        if extraction is None or extraction["validation_error"] is not None:
            self.last_trace[f"error"] = {
                "question": question,
                "prompt": leader_prompt,
                "response": leader_consulations,
                "error": extraction["validation_error"] if extraction else "unknown",
            }

            raise ValueError(f"Agent formatting error: {self.last_trace}")

        consulations = [
            (list(level1_dict.keys())[int(x["expert_id"] - 1)], x["question"]) for x in extraction["inquiries"]
        ]

        rebuttals = []
        for consul in consulations:
            answer, inquiry = consul
            if answer not in critics.keys():
                continue

            rebuttal_query = f"""{debate_history[answer]}\nPlease answer the question from the leader toward your support report in 1-3 sentences, do not change your stance:{inquiry}."""
            rebuttal = self._query_backbone(image, rebuttal_query, temperature=0.1)
            self.last_trace[f"level3_rebuttal_{answer}"] = rebuttal
            rebuttals.append(f"(Critic for {answer} - response)\n{rebuttal}")

        rebuttals = "(Expert Panel Response)\n" + "\n".join(rebuttals)
        level3_reports = critic_text + "[Leader Inquiries]" + leader_consulations + rebuttals

        leader_report_query = f"""
        [Response to your inquiries] {rebuttals} 

        [Task 2] You have received all critiques and the final responses to your inquiries. Your task is to render the final, binding verdict on this case. Your decision must be based on which hypothesis best survived the logical stress test. 
        
        [Adjudication Methodology] Strictly follow these steps in your thinking: 1. Global Review: Re-examine the complete record: the source evidence, the Critique Reports from each Critic Agent, your inquiries, and the Critics' final responses to those inquiries. 2. Compare Critique Impact: Your primary task is to compare the severity and impact of the flaws identified. Synthesize all information to determine which hypothesis, after rigorous scrutiny, best survived its dedicated critique. 3. Justify the Verdict: You must explicitly state why one hypothesis survived better than the other(s). Your final reasoning MUST be based on this direct comparison. 4. Render Final Verdict: Formulate your final, reasoned judgment, you can choose an overlooked choice when you are very confident after careful thinking. 
        
        [Strict Instruction] This is the final step. No further escalation is possible. 

        [Answer Format Constraints]
        - If the question type is "open": provide a concise but complete medical answer (no format restriction).
        - If the question type is "yes/no": the answer MUST be exactly either "yes" or "no".
        - If the question type is "multi-choice": the answer MUST be a single uppercase letter (e.g., A, B, C, D).
        
        [Strict Output Format] 
        #Question Type: <open / yes/no / multi-choice>
        #Reasoning: <A concise 2-4 sentence report comparing the critiques and justifying the final verdict.> 
        #Answer: <final answer strictly following the format constraint>
        """

        retry_count = 0
        while retry_count < MAX_RETRY:
            leader_final_report = self._query_backbone(image, leader_report_query, temperature=0.1)
            self.last_trace[f"level3_leader_final"] = leader_final_report
            extraction = self._extract_option(leader_final_report)

            if extraction["validation_error"] is None:
                break  # ✅ success

            retry_count += 1

        if extraction is None or extraction["validation_error"] is not None:
            self.last_trace[f"error"] = {
                "question": question,
                "prompt": leader_report_query,
                "response": leader_final_report,
                "error": extraction["validation_error"] if extraction else "unknown",
            }

            raise ValueError(f"Agent formatting error: {self.last_trace}")

        return leader_final_report, extraction

    def infer_vision_language(self, image, qs, image_size=None, temperature=None):

        self.reset()

        # ---------------------------
        # Level 1
        # ---------------------------

        level1_outputs = []
        level1_dict = defaultdict(list)

        try:
            for _ in range(2):
                response, extraction = self._single_expert_level1(qs, image, temperature=0.7, expert_id=_)
                option = extraction["answer"]

                level1_outputs.append(response)
                level1_dict[option].append(response)

            level1_report = "(Level-1 Initial Assessment Reports)\n" + "\n".join(
                [f"<Agent{i+1} {output}>" for i, output in enumerate(level1_outputs)]
            )
            latest_report = "" + level1_report

            # unanimous
            if len(level1_dict) == 1:
                response, extraction = self._level2_diagnosis(qs, image, level1_report)
                latest_report += "\n(Level-2 Extra Expert Check)\n" + f"<{response}>"

                level2_option = extraction["answer"]

                only_option = list(level1_dict.keys())[0]

                if level2_option == only_option:
                    return level2_option

            # ---------------------------
            # Level 3 (debate)
            # ---------------------------
            response, extraction = self._level3_diagnosis(qs, image, latest_report, level1_dict)
            level3_option = extraction["answer"]

            return level3_option

        except Exception as e:
            if "error" not in self.last_trace.keys():
                self.last_trace["error"] = {"system_message": str(e)}

            print(f"Trace history: {self.last_trace}\n" f"Error occurred: {e}\n" f"Switching to zero-shot mode.")
            # if any formatting error happens in any stage of the conversation, use zero-shot instead
            return self._query_backbone(image, qs, image_size=image_size, temperature=temperature)

    def _extract_option(self, response, question_type_pre=None):
        """
        Industrial-grade parser with auto-correction.
        Returns structured output + correction flags.
        """

        result = {
            "question_type": None,
            "reasoning": None,
            "answer": None,
            "valid_format": False,
            "corrected": False,
            "validation_error": None,
        }

        if not response or not isinstance(response, str):
            result["validation_error"] = "Empty response"
            return result

        text = response.strip()

        # --------------------------------------------------
        # STEP 1 — Extract raw fields (flexible matching)
        # --------------------------------------------------

        qtype_match = re.search(r"#?\s*Question\s*Type\s*:\s*(.+)", text, re.IGNORECASE)

        reasoning_match = re.search(
            r"#?\s*Reasoning\s*:\s*(.*?)\s*(?=#?\s*Answer\s*:|$)", text, re.IGNORECASE | re.DOTALL
        )

        answer_match = re.search(r"#?\s*Answer\s*:\s*(.+)", text, re.IGNORECASE)

        # Extract values if found
        question_type = qtype_match.group(1).strip().lower() if qtype_match else None
        reasoning = reasoning_match.group(1).strip() if reasoning_match else None
        answer = answer_match.group(1).strip() if answer_match else None

        # --------------------------------------------------
        # STEP 2 — Auto-correct Question Type
        # --------------------------------------------------

        if question_type:
            qt = question_type.lower()

            if "yes" in qt:
                question_type = "yes/no"
                result["corrected"] = True

            elif "multi" in qt or "choice" in qt:
                question_type = "multi-choice"
                result["corrected"] = True

            elif "open" in qt:
                question_type = "open"

            elif qt not in ["open", "yes/no", "multi-choice"]:
                # unknown → fallback later
                question_type = None

        # --------------------------------------------------
        # STEP 3 — Auto-detect Question Type if missing
        # --------------------------------------------------

        if not question_type and answer:

            # Detect yes/no
            if re.search(r"\b(yes|no)\b", answer, re.IGNORECASE):
                question_type = "yes/no"
                result["corrected"] = True

            # Detect multi-choice
            elif re.search(r"\b[A-Z]\b", answer):
                question_type = "multi-choice"
                result["corrected"] = True

            else:
                question_type = "open"
                result["corrected"] = True

        if question_type_pre is not None and question_type != question_type_pre:
            question_type = question_type_pre
            result["validation_error"] = "Detected question type does not match previous responses."

        # --------------------------------------------------
        # STEP 4 — Auto-correct Answer
        # --------------------------------------------------

        if answer and question_type:

            original_answer = answer

            if question_type == "yes/no":
                match = re.search(r"\b(yes|no)\b", answer, re.IGNORECASE)
                if match:
                    answer = match.group(1).lower()
                else:
                    result["validation_error"] = "Cannot auto-correct yes/no answer"
                    return result

            elif question_type == "multi-choice":
                # Extract first capital letter
                match = re.search(r"\b([A-Z])\b", answer.upper())
                if match:
                    answer = match.group(1)
                else:
                    result["validation_error"] = "Cannot auto-correct multi-choice answer"
                    return result

            elif question_type == "open":
                answer = answer.strip()

            if answer != original_answer:
                result["corrected"] = True

        # --------------------------------------------------
        # STEP 5 — Final Validation
        # --------------------------------------------------

        if not reasoning:
            # fallback: remove header sections
            reasoning = text
            result["corrected"] = True

        result["question_type"] = question_type
        result["reasoning"] = reasoning
        result["answer"] = answer

        if question_type and answer:
            result["valid_format"] = True
        else:
            result["validation_error"] = "Missing required fields"

        return result

    def _extract_inquiries(self, response: str):
        """
        Relaxed extractor for Lead Adjudicator output.

        Accepts formats like:
        @ To Expert 1: Question...
        @ To Expert 2: Question...

        or
        @ To Expert 1 who reviews Agent1: Question...
        """

        results = []

        if not response or not isinstance(response, str):
            return {
                "inquiries": [],
                "valid_format": False,
                "validation_error": "Empty response",
            }

        text = response.strip()

        # Relaxed pattern:
        # - Start with @
        # - Anything until first colon
        # - Everything after colon until next @ or end
        pattern = re.compile(
            r"@\s*(.+?)\s*:\s*(.+?)(?=\s*@|\Z)",
            re.IGNORECASE | re.DOTALL,
        )

        matches = pattern.findall(text)

        if len(matches) < 2:
            return {
                "inquiries": [],
                "valid_format": False,
                "validation_error": "Less than two valid inquiries found",
            }

        for header, question_text in matches:
            # Try extracting expert number if present
            expert_match = re.search(r"Expert\s*(\d+)", header, re.IGNORECASE)
            expert_id = int(expert_match.group(1)) if expert_match else None

            results.append(
                {
                    "expert_id": expert_id,  # None if not found
                    "header": header.strip(),
                    "question": question_text.strip(),
                }
            )

        return {
            "inquiries": results,
            "valid_format": True,
            "validation_error": None,
        }
