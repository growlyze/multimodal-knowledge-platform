FROM ghcr.io/open-webui/open-webui:v0.11.4

# Product name shown in the browser title and UI. compose.yml passes APP_NAME
# from .env as this build arg (and as the WEBUI_NAME runtime variable), so
# each client deployment carries its own branding.
ARG WEBUI_NAME="Growlyze Multimodal AI Chat"
ENV WEBUI_NAME=${WEBUI_NAME}

# WEBUI_NAME alone appends "(Open WebUI)" after the name — patch that out so
# the title reads as just the configured product name.
RUN sed -i "/if WEBUI_NAME != 'Open WebUI':/,+1d" /app/backend/open_webui/env.py

# Per-client logo / favicon / CSS: drop the files in this folder and uncomment.
# COPY favicon.svg /app/build/static/favicon.svg
# COPY favicon.png /app/build/static/favicon.png
# COPY custom.css  /app/build/static/custom.css
