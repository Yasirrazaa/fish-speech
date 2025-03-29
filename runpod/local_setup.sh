#!/bin/bash

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Fish Speech Local Setup${NC}"
echo "========================"

# Make scripts executable
echo -e "\n${GREEN}Making scripts executable...${NC}"
chmod +x cleanup.sh setup.sh test_local.sh

# Run cleanup
echo -e "\n${GREEN}Running cleanup...${NC}"
./cleanup.sh

# Check Docker installation
echo -e "\n${GREEN}Checking Docker...${NC}"
if ! command -v docker &> /dev/null; then
    echo -e "${RED}Error: Docker is not installed${NC}"
    echo "Please install Docker and NVIDIA Container Toolkit"
    exit 1
fi

# Check NVIDIA Docker
echo -e "\n${GREEN}Checking NVIDIA Docker...${NC}"
if ! docker info | grep -i nvidia &> /dev/null; then
    echo -e "${RED}Error: NVIDIA Docker runtime not found${NC}"
    echo "Please install NVIDIA Container Toolkit:"
    echo "https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
    exit 1
fi

# Check models
echo -e "\n${GREEN}Checking model files...${NC}"
MODEL_DIR="../checkpoints"

check_model() {
    local model_path="$1"
    local model_name="$2"
    if [ ! -d "$model_path" ]; then
        echo -e "${YELLOW}$model_name not found. Downloading...${NC}"
        huggingface-cli download "$model_name" --local-dir "$model_path"
    else
        echo -e "${GREEN}$model_name found${NC}"
    fi
}

mkdir -p "$MODEL_DIR"
check_model "$MODEL_DIR/fish-agent-v0.1-3b" "fishaudio/fish-agent-v0.1-3b"
check_model "$MODEL_DIR/fish-speech-1.4" "fishaudio/fish-speech-1.4"

# Install test dependencies
echo -e "\n${GREEN}Installing test dependencies...${NC}"
pip install -q runpod soundfile numpy

# Run local test
echo -e "\n${GREEN}Running local test...${NC}"
./test_local.sh

echo -e "\n${GREEN}Setup complete!${NC}"
echo -e "You can now proceed to deploy on RunPod following the README.md"
