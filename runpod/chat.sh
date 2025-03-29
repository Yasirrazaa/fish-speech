#!/bin/bash

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Make test_client.py executable
chmod +x test_client.py

# Create outputs directory
mkdir -p outputs

# Function to show usage
show_usage() {
    echo -e "${YELLOW}Fish Speech Chat Client${NC}"
    echo "Usage:"
    echo "  ./chat.sh [options]"
    echo
    echo "Options:"
    echo "  text       Simple text chat"
    echo "  voice      Chat with voice cloning"
    echo "  convo     Continue conversation"
    echo
    echo "Examples:"
    echo "  ./chat.sh text \"Hello, how are you?\""
    echo "  ./chat.sh voice \"Tell me a story\" reference.wav"
    echo "  ./chat.sh convo \"Tell me more\" story-1"
}

case $1 in
    text)
        if [ -z "$2" ]; then
            echo -e "${RED}Error: Please provide a message${NC}"
            echo "Usage: ./chat.sh text \"Your message\""
            exit 1
        fi
        echo -e "${GREEN}Sending text chat...${NC}"
        ./test_client.py --message "$2"
        ;;
        
    voice)
        if [ -z "$2" ] || [ -z "$3" ]; then
            echo -e "${RED}Error: Please provide both message and audio file${NC}"
            echo "Usage: ./chat.sh voice \"Your message\" reference.wav"
            exit 1
        fi
        if [ ! -f "$3" ]; then
            echo -e "${RED}Error: Audio file not found: $3${NC}"
            exit 1
        fi
        echo -e "${GREEN}Sending voice chat...${NC}"
        ./test_client.py --message "$2" --reference-audio "$3"
        ;;
        
    convo)
        if [ -z "$2" ] || [ -z "$3" ]; then
            echo -e "${RED}Error: Please provide message and conversation ID${NC}"
            echo "Usage: ./chat.sh convo \"Your message\" conversation-id"
            exit 1
        fi
        echo -e "${GREEN}Continuing conversation...${NC}"
        ./test_client.py --message "$2" --conversation-id "$3"
        ;;
        
    *)
        show_usage
        ;;
esac

# Show output files if they exist
if [ -f "outputs/response.txt" ]; then
    echo -e "\n${YELLOW}Latest Response:${NC}"
    cat outputs/response.txt
fi

if [ -f "outputs/response.wav" ]; then
    echo -e "\n${YELLOW}Audio response saved to:${NC} outputs/response.wav"
fi
