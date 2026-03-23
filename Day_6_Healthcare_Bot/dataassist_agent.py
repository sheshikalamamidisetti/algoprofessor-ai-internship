# ============================================================
# DATAASSIST ANALYTICS AGENT
# Day 6 Healthcare Bot: DataAssist Analytics Agent
# Author: Sheshikala
# Topic: Full DataAssist agent combining all Day 6 components
# ============================================================

# This is the capstone file for Day 6. It combines all five
# components built in this folder into a single intelligent
# analytics agent:
#   openai_client    - LLM generation with OpenAI or Claude
#   function_calling - precise data retrieval with real tools
#   pydantic_schemas - structured and validated outputs
#   memory_manager   - multi-turn conversational memory
#   report_generator - automated formatted report generation
# The DataAssistAgent accepts natural language questions about
# the Titanic dataset, classifies each question into one of
# five intent categories, and routes it to the most appropriate
# handler. Conversation history is maintained across all turns
# so the agent remembers what was discussed earlier in the
# same session.

from datetime import datetime

from data_loader      import load_titanic, summarize_dataframe
from openai_client    import OpenAIClient
from claude_client    import ClaudeClient
from function_calling import FunctionCallingAgent, FunctionCallDispatcher
from memory_manager   import ConversationManager, BufferMemory
from report_generator import ReportGenerator, compute_report_statistics


# ============================================================
# INTENT CLASSIFIER
# ============================================================

def classify_intent(question):
    """
    Classifies the user question into one of five categories
    to route it to the correct handler in the agent.

    Categories:
        report       - user wants a full formatted report
        statistic    - user wants a specific computed metric
        comparison   - user wants to compare groups
        prediction   - user wants a prediction or assessment
        conversation - general question or follow-up turn

    Parameters:
        question : string

    Returns:
        string intent category name
    """
    q = question.lower()

    report_phrases = ["generate report", "write report", "create report",
                      "full report", "complete report", "analysis report"]
    if any(phrase in q for phrase in report_phrases):
        return "report"

    statistic_words = ["survival rate", "percentage", "how many", "average",
                       "mean", "median", "statistics", "count", "missing",
                       "correlation", "std", "distribution"]
    if any(word in q for word in statistic_words):
        return "statistic"

    comparison_words = ["compare", "versus", "vs", "difference between",
                        "better than", "higher than", "lower than",
                        "by gender", "by class", "by sex", "which group"]
    if any(word in q for word in comparison_words):
        return "comparison"

    prediction_words = ["predict", "would", "survive", "chances",
                        "likelihood", "probability", "would this passenger",
                        "if a passenger"]
    if any(word in q for word in prediction_words):
        return "prediction"

    return "conversation"


# ============================================================
# DATAASSIST AGENT CLASS
# ============================================================

class DataAssistAgent:
    """
    Full DataAssist Analytics Agent for Titanic passenger data.

    Routes each user question to the appropriate handler:
        report intent      -> ReportGenerator.build_full_report()
        statistic intent   -> FunctionCallingAgent.answer()
        comparison intent  -> FunctionCallingAgent.answer()
        prediction intent  -> LLM with full dataset context
        conversation intent -> ConversationManager.chat()

    Maintains BufferMemory across all interactions in a session
    so the agent can reference earlier turns in the conversation.

    Parameters:
        use_claude : bool, use Claude instead of OpenAI if True
    """

    def __init__(self, use_claude=False):
        self.df      = load_titanic()
        self.summary = summarize_dataframe(self.df)
        self.stats   = compute_report_statistics(self.df)

        if use_claude:
            self.client      = ClaudeClient()
            self.client_name = "Claude"
        else:
            self.client      = OpenAIClient()
            self.client_name = "GPT"

        self.memory       = BufferMemory(max_tokens=3000)
        self.conversation = ConversationManager(
            self.memory, self.client, self.summary
        )
        self.func_agent   = FunctionCallingAgent()
        self.reporter     = ReportGenerator(self.client, self.client_name)

        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.turn_count = 0

        print("=" * 55)
        print("DataAssist Analytics Agent initialized")
        print("Session ID : " + self.session_id)
        print("LLM Client : " + self.client_name)
        print("Dataset    : Titanic (" + str(len(self.df)) + " records)")
        print("=" * 55)

    def ask(self, question):
        """
        Main entry point for all user questions. Classifies the
        intent and routes to the correct handler automatically.

        Parameters:
            question : string, the user question

        Returns:
            string, the agent response
        """
        self.turn_count += 1
        intent = classify_intent(question)

        print("\n[Turn " + str(self.turn_count) + "] Intent: " + intent)
        print("User: " + question)
        print("-" * 45)

        if intent == "report":
            response = self._handle_report(question)
        elif intent in ("statistic", "comparison"):
            response = self._handle_statistic(question)
        elif intent == "prediction":
            response = self._handle_prediction(question)
        else:
            response = self._handle_conversation(question)

        print("Agent: " + response[:300])
        return response

    def _handle_report(self, question):
        """Generates a full structured report and returns a summary."""
        print("Routing to ReportGenerator...")
        try:
            report = self.reporter.build_full_report()
            response = (
                "I have generated a full Titanic survival analysis report. "
                "Here is a preview of the executive summary:\n\n" +
                report[:500] + "\n\n[Full report generated successfully]"
            )
            self.memory.add_user_message(question)
            self.memory.add_assistant_message(response)
            return response
        except Exception as e:
            return "Report generation failed: " + str(e)

    def _handle_statistic(self, question):
        """Uses function calling to retrieve precise statistics."""
        print("Routing to FunctionCallingAgent...")
        try:
            return self.func_agent.answer(question)
        except Exception as e:
            print("Function calling failed, falling back to conversation: " + str(e))
            return self._handle_conversation(question)

    def _handle_prediction(self, question):
        """Uses LLM with full dataset context for predictions."""
        print("Routing to prediction handler...")
        prompt = (
            "Using these Titanic survival statistics, answer the prediction "
            "question with specific step-by-step reasoning:\n\n"
            "Statistics:\n" + self.summary + "\n\n"
            "Question: " + question
        )
        response = self.client.complete(prompt)
        self.memory.add_user_message(question)
        self.memory.add_assistant_message(response)
        return response

    def _handle_conversation(self, question):
        """Uses conversational memory for general questions."""
        print("Routing to ConversationManager...")
        return self.conversation.chat(question)

    def show_session_summary(self):
        """Displays a summary of the current session."""
        print("\n-- Session Summary --")
        print("Session ID   : " + self.session_id)
        print("Total turns  : " + str(self.turn_count))
        print("Memory state : " + self.memory.summary())
        print("Dataset      : " + str(self.stats.get("total_records", "N/A")) +
              " records, " + str(self.stats.get("overall_survival_rate", "N/A")) +
              "% survival rate")

    def export_session(self):
        """Returns the full session history as a dict."""
        return {
            "session_id" : self.session_id,
            "turn_count" : self.turn_count,
            "client"     : self.client_name,
            "history"    : self.conversation.export_history()
        }


# ============================================================
# DEMO
# ============================================================

def run_demo():
    print("=" * 55)
    print("DATAASSIST ANALYTICS AGENT DEMO")
    print("=" * 55)

    agent = DataAssistAgent(use_claude=False)

    demo_questions = [
        "What is the survival rate for female passengers?",
        "How does survival rate compare between passenger classes?",
        "What are the age statistics in this dataset?",
        "Are there any missing values I should know about?",
        "Would a first-class female passenger aged 30 likely survive?",
        "Based on what you just said, which factor matters most?",
        "Can you generate a complete analysis report?",
        "What was the most important insight from our conversation?"
    ]

    for question in demo_questions:
        agent.ask(question)

    agent.show_session_summary()

    print("\n-- Session export preview (first 3 turns) --")
    session_data = agent.export_session()
    for turn in session_data["history"][:6]:
        print(turn["role"].upper() + ": " + turn["content"][:100])

    print("\n-- DataAssist Analytics Agent demo complete --")


if __name__ == "__main__":
    run_demo()
