#!/bin/bash
###############################################################################
# EXACT MPPI ROS2 Workspace - Dependency Installation Script
#
# This script installs system-level dependencies for the ROS2 bridge workspace,
# including ROS2 packages and C++ libraries.
#
# Note: The exact_mppi Python environment is managed separately from this script.
#
# Usage:
#   chmod +x setup.sh
#   ./setup.sh
#
###############################################################################

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}EXACT MPPI ROS2 Workspace Setup${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# Check if ROS2 is installed
echo -e "${YELLOW}[1/3] Checking ROS2 installation...${NC}"
if [ -z "$ROS_DISTRO" ]; then
    echo -e "${YELLOW}ROS2 not sourced. Attempting to source ROS2 Kilted...${NC}"
    if [ -f "/opt/ros/kilted/setup.bash" ]; then
        source /opt/ros/kilted/setup.bash
        echo -e "${GREEN}✓ ROS2 Kilted sourced successfully${NC}"
    elif [ -f "/opt/ros/humble/setup.bash" ]; then
        source /opt/ros/humble/setup.bash
        echo -e "${GREEN}✓ ROS2 Humble sourced successfully${NC}"
    else
        echo -e "${RED}✗ ROS2 not found. Please install ROS2 first.${NC}"
        echo -e "  Visit: https://docs.ros.org/en/kilted/Installation.html"
        exit 1
    fi
else
    echo -e "${GREEN}✓ ROS2 $ROS_DISTRO detected${NC}"
fi
echo ""

# Install ROS2 dependencies
echo -e "${YELLOW}[2/3] Installing ROS2 package dependencies...${NC}"
sudo apt update
sudo apt install -y \
    ros-$ROS_DISTRO-tf-transformations \
    ros-$ROS_DISTRO-tf2-tools \
    ros-$ROS_DISTRO-tf2-ros \
    ros-$ROS_DISTRO-tf2-geometry-msgs \
    ros-$ROS_DISTRO-nav-msgs \
    ros-$ROS_DISTRO-sensor-msgs \
    ros-$ROS_DISTRO-geometry-msgs \
    ros-$ROS_DISTRO-visualization-msgs \
    ros-$ROS_DISTRO-rviz2 \
    python3-colcon-common-extensions

echo -e "${GREEN}✓ ROS2 dependencies installed${NC}"
echo ""

# Install C++ dependencies (for ddr_minimal_sim)
echo -e "${YELLOW}[3/3] Installing C++ dependencies...${NC}"
sudo apt install -y \
    libeigen3-dev \
    libyaml-cpp-dev \
    build-essential \
    cmake

echo -e "${GREEN}✓ C++ dependencies installed${NC}"
echo ""

# Summary
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}System Dependencies Installed!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "${BLUE}IMPORTANT: Python Dependencies${NC}"
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "This script does NOT install the exact_mppi Python package or its JAX stack."
echo -e "Build-time ROS Python helpers are installed automatically by ./build.sh."
echo ""
echo -e "Install your controller environment separately before running the bridge:"
echo ""
echo -e "  ${YELLOW}source ./EXACT-mppi/.exact_mppi/bin/activate${NC}"
echo -e "  ${YELLOW}python -m pip install -e ./EXACT-mppi/EXACT_MPPI_core${NC}"
echo ""
echo -e "The ROS2 bridge also expects ROS Python packages such as rclpy to be available"
echo -e "in the runtime environment you use to launch nodes."
echo ""
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo -e "Next steps:"
echo -e "  ${YELLOW}1.${NC} Install Python dependencies"
echo -e "     Activate your exact_mppi environment and install EXACT_MPPI_core"
echo ""
echo -e "  ${YELLOW}2.${NC} Build the workspace:"
echo -e "     ${YELLOW}./build.sh${NC}"
echo -e "     or"
echo -e "     ${YELLOW}colcon build --symlink-install${NC}"
echo ""
echo -e "  ${YELLOW}3.${NC} Source the workspace:"
echo -e "     ${YELLOW}source install/setup.bash${NC}"
echo ""
echo -e "  ${YELLOW}4.${NC} Run the demo:"
echo -e "     ${YELLOW}ros2 launch exact_mppi_jax sim_corridor_external_ref_launch.py${NC}"
echo ""
echo -e "${GREEN}For more information, see README.md${NC}"
echo ""
