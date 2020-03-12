#! /bin/bash
mkdir dataset/"$1"
read -p "Enter search query : " var
echo "$var"
python search_bing_api.py --query "$var" --output dataset/"$1"


