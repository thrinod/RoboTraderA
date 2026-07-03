# Upstox Proxy Server Deployment Guide

Your repository contains [upstox_proxy_server.py](upstox_proxy_server.py), which is designed to run on AWS, accept incoming order placements, and forward them directly to Upstox using the AWS public IP (forcing IPv4).

Follow these steps to deploy, activate, and configure the proxy for automatic restarts:

---

## Step 1: Prepare the AWS EC2 Instance
1. Spin up a basic Ubuntu Server EC2 instance.
2. Assign an **Elastic IP** to the instance (to ensure it has a static public IP that won't change).
3. **Whitelist this Elastic IP** in your Upstox Developer Portal.
4. **AWS Security Group Configuration**: Open the port you want the proxy to listen on (e.g. `8080`) to allow incoming requests from your local server's IP address.

---

## Step 2: Set Up and Run the Proxy on AWS

1. **SSH into your AWS instance**:
   ```bash
   ssh -i your-key.pem ubuntu@your-aws-elastic-ip
   ```
2. **Install Python & Pip**:
   ```bash
   sudo apt update
   sudo apt install python3-pip python3-venv -y
   ```
3. **Transfer the script** (`upstox_proxy_server.py`) to the AWS instance.
4. **Create a virtual environment & install dependencies**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install fastapi uvicorn httpx
   ```

---

## Step 3: Test the Proxy Health
* Run uvicorn temporarily:
  ```bash
  uvicorn upstox_proxy_server:app --host 0.0.0.0 --port 8080
  ```
* Open your browser and navigate to:
  `http://<YOUR_AWS_ELASTIC_IP>:8080/check-ip`
* It should return a success message confirming the outgoing AWS IP address seen by Upstox.

---

## Step 4: Configure Auto-Start on System Boot (Persistence)

To make sure the proxy server starts automatically whenever the AWS instance boots or restarts, you can use **PM2** or **Systemd**.

### Option A: Using PM2 (Recommended)
1. **Install Node.js & PM2**:
   ```bash
   sudo apt install nodejs npm -y
   sudo npm install -g pm2
   ```
2. **Start the Proxy under PM2**:
   ```bash
   # Run the script using the virtual environment interpreter
   pm2 start upstox_proxy_server.py --name "upstox-proxy" --interpreter venv/bin/python -- --port 8080
   ```
3. **Configure PM2 to start on boot**:
   ```bash
   pm2 startup
   ```
   *This command will output a specific command starting with `sudo env PATH=...`. **Copy and run that output command** on your terminal to register the startup hook.*
4. **Save current running processes list**:
   ```bash
   pm2 save
   ```
   *This saves the list of running processes so they reload automatically after reboot.*

---

### Option B: Using a Linux Systemd Service (Alternative)
1. **Create a systemd service file**:
   ```bash
   sudo nano /etc/systemd/system/upstox-proxy.service
   ```
2. **Paste the following configuration** (adjust paths as needed):
   ```ini
   [Unit]
   Description=Upstox API Proxy Server
   After=network.target

   [Service]
   User=ubuntu
   WorkingDirectory=/home/ubuntu
   ExecStart=/home/ubuntu/venv/bin/uvicorn upstox_proxy_server:app --host 0.0.0.0 --port 8080
   Restart=always
   RestartSec=5

   [Install]
   WantedBy=multi-user.target
   ```
3. **Reload systemd configurations and enable on boot**:
   ```bash
   sudo systemctl daemon-reload
   sudo systemctl enable upstox-proxy.service
   sudo systemctl start upstox-proxy.service
   ```

---

## Step 5: Configure Your Local RoboTrader Backend
To route your local orders through the AWS Proxy:
1. In your local `backend/.env` file, add or update the Upstox API URL to point to your AWS proxy server:
   ```env
   UPSTOX_API_BASE_URL=http://<YOUR_AWS_ELASTIC_IP>:8080
   ```
2. In your backend Upstox connection service, ensure all requests fetch from `UPSTOX_API_BASE_URL` instead of hitting `https://api.upstox.com` directly.
