from pyboy import PyBoy
import time

def test_pyboy_api():
    rom_path = "/Users/danieltomaro/Documents/Projects/Pokemon/ROMs/Pokemon Red.gb"
    pyboy = PyBoy(rom_path, window="SDL2")
    
    # Define button constants
    BUTTON_A = 0x01
    BUTTON_B = 0x02
    BUTTON_SELECT = 0x04
    BUTTON_START = 0x08
    BUTTON_RIGHT = 0x10
    BUTTON_LEFT = 0x20
    BUTTON_UP = 0x40
    BUTTON_DOWN = 0x80
    
    try:
        print("\nTesting basic controls...")
        
        # Run for a few frames
        for _ in range(100):
            pyboy.tick()
        
        print("\nTesting button press...")
        
        # Test button press sequence
        test_buttons = [
            ("A", BUTTON_A),
            ("RIGHT", BUTTON_RIGHT),
            ("START", BUTTON_START)
        ]
        
        for button_name, button_value in test_buttons:
            print(f"\nTesting {button_name} button...")
            # Press
            pyboy.send_input(button_value)
            pyboy.tick()
            time.sleep(0.5)
            # Release
            pyboy.send_input(0)  # Release all buttons
            pyboy.tick()
            time.sleep(0.5)
        
        # Test memory access
        print("\nTesting memory access...")
        x_pos = pyboy.memory[0xD362]
        y_pos = pyboy.memory[0xD361]
        print(f"Player position: ({x_pos}, {y_pos})")
        
        print("\nTest complete!")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        pyboy.stop()

if __name__ == "__main__":
    test_pyboy_api()