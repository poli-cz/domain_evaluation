#!/bin/bash
cp ./models ./web_app/ -r
cp ./src ./web_app/ -r

echo "Backend configured"

cd ./web_app
npm i

echo "Frontend configured"
