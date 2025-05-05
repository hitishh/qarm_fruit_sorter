#THIS IS FOR COMPLETELY MANUAL TELOPERATION, EACH JOINT IS CONTROLLED MANUALLY
import os
import sys
import numpy as np
import pygame
sys.path.append(r"C:\Users\hitis\Documents\Quanser\0_libraries\python")
from pal.products.qarm import QArm

def main():
    # Initialize pygame with minimal setup
    pygame.init()
    screen = pygame.display.set_mode((300, 200), pygame.HWSURFACE | pygame.DOUBLEBUF)
    pygame.display.set_caption('QArm Control')
    pygame.event.set_allowed([pygame.QUIT, pygame.KEYDOWN])  # Filter only needed events

    # Constants (pre-calculated)
    ANGLE_STEP = np.deg2rad(5)  # ~0.087 radians
    GRIPPER_STEP = 0.1

    # Joint angle limits (radians, pre-converted)
    pi = np.pi
    JOINT_LIMITS = np.array([
        [-pi, pi],   # joint_1: ~(-180°, 180°)
        [-pi, pi],   # joint_2: ~(-180°, 180°)
        [-pi, pi],   # joint_3: ~(-180°, 180°)
        [-pi, pi]    # joint_4: ~(-180°, 180°)
    ])

    # Control mappings (key: (joint_index, direction))
    KEY_MAPPINGS = {
        pygame.K_a: (0, 1),    # Joint 1 increase
        pygame.K_q: (0, -1),   # Joint 1 decrease
        pygame.K_w: (1, 1),    # Joint 2 increase
        pygame.K_s: (1, -1),   # Joint 2 decrease
        pygame.K_e: (2, 1),    # Joint 3 increase
        pygame.K_d: (2, -1),   # Joint 3 decrease
        pygame.K_r: (3, 1),    # Joint 4 increase
        pygame.K_f: (3, -1),   # Joint 4 decrease
        pygame.K_z: (-1, 1),   # Gripper open
        pygame.K_x: (-1, -1)   # Gripper close
    }

    # Initial state
    joint_angles = np.zeros(4, dtype=np.float64)
    gripper = 0.5
    led_cmd = np.array([0, 1, 0], dtype=np.float64)

    # Print controls once
    print(
        "--- QArm Control Guide ---\n"
        "Joint 1: A (increase), Q (decrease)\n"
        "Joint 2: W (increase), S (decrease)\n"
        "Joint 3: E (increase), D (decrease)\n"
        "Joint 4: R (increase), F (decrease)\n"
        "Gripper: Z (open), X (close)\n"
        "Exit: ESC key\n"
        "---------------------------\n"
    )

    with QArm(hardware=0) as myArm:
        # Move to home position
        print("Moving to home position...")
        myArm.read_write_std(phiCMD=joint_angles, gprCMD=gripper, baseLED=led_cmd)
        pygame.time.delay(1000)  # More efficient than time.sleep()
        print("Reached home.")

        running = True
        clock = pygame.time.Clock()  # For controlling frame rate
        
        while running and myArm.status:
            clock.tick(30)  # Limit to 30 FPS to reduce CPU usage
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        print("Exiting...")
                        running = False
                        continue
                    
                    # Handle movement commands
                    if event.key in KEY_MAPPINGS:
                        joint_idx, direction = KEY_MAPPINGS[event.key]
                        
                        if joint_idx >= 0:  # Joint movement
                            joint_angles[joint_idx] = np.clip(
                                joint_angles[joint_idx] + direction * ANGLE_STEP,
                                JOINT_LIMITS[joint_idx, 0],
                                JOINT_LIMITS[joint_idx, 1]
                            )
                        else:  # Gripper movement
                            gripper = np.clip(
                                gripper + direction * GRIPPER_STEP,
                                0.0, 1.0
                            )
                        
                        # Send command to arm
                        myArm.read_write_std(
                            phiCMD=joint_angles,
                            gprCMD=gripper,
                            baseLED=led_cmd
                        )
                        
                        # Print status (less frequent updates might be better)
                        print(
                            f"Joints (rad): {np.around(joint_angles, 4)}\n"
                            f"Joints (deg): {np.around(np.rad2deg(joint_angles), 2)}\n"
                            f"Gripper: {gripper:.2f}\n"
                        )

    pygame.quit()

if __name__ == "__main__":
    main()
