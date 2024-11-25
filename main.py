from pokemon_ai import PokemonAI

if __name__ == "__main__":
    try:
        print("Creating PokemonAI instance...")
        ai = PokemonAI()
        print("Starting continuous training process...")
        ai.train(steps_per_save=10000)  # Continuous training with progress saved every 10,000 steps
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    except Exception as e:
        print(f"Error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("Cleaning up...")
        if 'ai' in locals() and hasattr(ai, 'pyboy'):
            ai.pyboy.stop()
        print("Cleanup complete")
