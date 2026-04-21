"""Interactive entrypoint for BioProAgent.

This module keeps a clean, user-facing CLI while reusing the core runtime
implementation from `main_evaluate.py`.
"""

from main_evaluate import ProAgent, SessionState


class ProAgentV3(ProAgent):
    """Interactive runtime wrapper around the core ProAgent implementation."""

    def __init__(self, eval_mode: bool = False):
        super().__init__(eval_mode=eval_mode)

    def run_session(self) -> None:
        """Start an interactive chat session."""
        state = self._create_session()

        print("\n" + "=" * 60)
        print("BioProAgent Interactive Session")
        print("=" * 60)
        print("Commands: `quit` / `exit`, `show_state`, `clear`\n")

        while True:
            try:
                user_query = input("You: ").strip()
                if not user_query:
                    continue

                if user_query.lower() in {"quit", "exit", "q"}:
                    print("Bye.")
                    break

                if user_query.lower() == "show_state":
                    self._print_state(state)
                    continue

                if user_query.lower() == "clear":
                    state = self._create_session()
                    print("Session reset.")
                    continue

                response = self.process_query(user_query, state)
                print(f"\nBioProAgent: {response}\n")

            except KeyboardInterrupt:
                print("\nBye.")
                break
            except Exception as exc:
                print(f"Error: {exc}")

    def _print_state(self, state: SessionState) -> None:
        """Print a concise session debug snapshot."""
        print("\n" + "=" * 40)
        print("Current Session State")
        print("=" * 40)
        print(f"Session ID: {state.session_id}")
        print(f"Experiment Context: {state.global_exp_context[:120]}")
        print(f"Has Draft: {bool(state.current_draft)}")
        print(f"M_episodic Steps: {len(state.mem_episodic)}")
        print(f"M_work Keys: {list(state.mem_work.keys())}")
        print("=" * 40 + "\n")


if __name__ == "__main__":
    ProAgentV3().run_session()
