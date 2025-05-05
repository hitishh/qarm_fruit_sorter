#THIS IS FOR SEMI AUTONOMOUS TELOPERATION, THE LOCATION OF THE END EFFECTOR IS
#SPECIFIED BY PRE-DEFINED LOCATIONS OR COORDINATES, USING IK THE ROBOT IS MOVED
#TO SPECIFIED LOCATION, GRIPPER IS CONTROLLED WITH PYGAMES
import os
import sys
import numpy as np
import time
import pygame

# Add the path to the QArm libraries and the IK script
sys.path.append(r"C:\Users\hitis\Documents\Quanser\0_libraries\python")
sys.path.append(r"C:\Users\hitis\Documents\Quanser\5_research\qarm\basic")

from pal.products.qarm import QArm
from hal.products.qarm import QArmUtilities

def print_instructions():
    print(
        "--- QArm Control Instructions ---\n"
        "Enter coordinates (x, y, z) or use predefined positions.\n"
        "Gripper Control via Pygame Keyboard:\n"
        "- Press 'c' to close the gripper\n"
        "- Press 'o' to open the gripper\n"
        "- Press 'q' to continue\n"
        "--------------------------------------------\n"
        "Joint angle limits (rad): J1: [-π, π], J2: [-π/2, π/2], J3: [0, π], J4: [-π/2, π/2]\n"
        "Workspace limits (approx): x: (-0.8, 0.8), y: (-0.8, 0.8), z: (0.0, 0.8)\n"
    )

def validate_coordinates(x, y, z):
    x_limit = (-0.8, 0.8)
    y_limit = (-0.8, 0.8)
    z_limit = (0.0, 0.8)
    return x_limit[0] <= x <= x_limit[1] and y_limit[0] <= y <= y_limit[1] and z_limit[0] <= z <= z_limit[1]

def check_ik_solution_validity(phiCmd):
    return not np.any(np.isnan(phiCmd))

def control_gripper(gripCmd, myArm, phiCmd, led_cmd):
    print("\n--- Gripper Control ---\nPress 'c' to close, 'o' to open, 'q' to continue\n")
    
    pygame.init()
    screen = pygame.display.set_mode((400, 200))
    pygame.display.set_caption('Gripper Control')
    font = pygame.font.SysFont(None, 32)

    running = True
    while running:
        screen.fill((30, 30, 30))
        text = font.render(f"Gripper: {gripCmd:.2f} | Press 'q' to continue", True, (200, 200, 200))
        screen.blit(text, (30, 80))
        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return gripCmd
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False
                elif event.key == pygame.K_c:
                    gripCmd = min(gripCmd + 0.2, 1.0)
                    print(f"Gripper opened to: {gripCmd:.2f}")
                elif event.key == pygame.K_o:
                    gripCmd = max(gripCmd - 0.2, 0.0)
                    print(f"Gripper closed to: {gripCmd:.2f}")
                myArm.read_write_std(phiCMD=phiCmd, gprCMD=gripCmd, baseLED=led_cmd)

        time.sleep(0.1)

    pygame.quit()
    return gripCmd

def main():
    led_cmd = np.array([0, 1, 0], dtype=np.float64)
    np.set_printoptions(precision=2, suppress=True)

    print_instructions()

    positions = {
        "R_tom": [-0.25, -0.4, 0.05],
        "R_ban": [-0.2, -0.5, 0.05],
        "R_str": [-0.3, -0.3, 0.05],
        "U_tom": [-0.35, -0.1, 0.05],
        "U_ban": [-0.42, 0.1, 0.05],
        "U_str": [-0.4, 0.3, 0.05],
        "Rott": [-0.1, 0.4, 0.05],
        "Pick_pose": [0.65, 0, 0.30],
        "home": [0.40, 0, 0.30]
    }

    with QArm(hardware=0) as myArm:
        myArmUtilities = QArmUtilities()
        gripCmd = 0.0

        while myArm.status:
            choice = input("Use predefined position? (yes/no): ").strip().lower()
            if choice == 'yes':
                print("Available positions:")
                for key in positions:
                    print(f"- {key}: {positions[key]}")
                pos_choice = input("Enter position name: ").strip()
                if pos_choice in positions:
                    positionCmd = np.array(positions[pos_choice])
                else:
                    print("Invalid position. Try again.")
                    continue
            else:
                try:
                    coords = input("Enter x y z: ").split()
                    if len(coords) != 3:
                        print("Enter exactly 3 values.")
                        continue
                    x, y, z = map(float, coords)
                    if not validate_coordinates(x, y, z):
                        print("Coordinates out of bounds.")
                        continue
                    positionCmd = np.array([x, y, z])
                except ValueError:
                    print("Invalid input.")
                    continue

            gamma = 0
            allPhi, phiCmd = myArmUtilities.qarm_inverse_kinematics(positionCmd, gamma, myArm.measJointPosition[0:4])
            if not check_ik_solution_validity(phiCmd):
                print("Unreachable position.")
                continue

            location, _ = myArmUtilities.qarm_forward_kinematics(np.append(phiCmd, gamma))
            phiCmd_deg = np.rad2deg(phiCmd)

            print(f"Target (x, y, z): {positionCmd}")
            print(f"Actual position: {location}")
            print(f"Joint angles (deg): {np.around(phiCmd_deg, 2)}")

            myArm.read_write_std(phiCMD=phiCmd, gprCMD=gripCmd, baseLED=led_cmd)
            gripCmd = control_gripper(gripCmd, myArm, phiCmd, led_cmd)
            time.sleep(0.5)

    print("Program ended.")

if __name__ == "__main__":
    main()
