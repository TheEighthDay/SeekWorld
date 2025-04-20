## Data from Google Maps

The diversity and difficulty levels of data play a critical role in reinforcement learning (RL). Therefore, our dataset exhibits two distinct advantages: 
1) globally diverse sample distribution;
2) better image-label pairs specifically curated for rule-based reinforcement learning;

To ensure comprehensive geographical diversity, we strategically collected street view images from 171 countries, categorized by urbanization levels across four principal domains: iconic landmark surroundings (Eiffel Tower, Statue of Liberty), metropolitan areas of global cities, rural/suburban regions with various geographical features (forests, grasslands, snow mountains, beaches), and user-contributed human-centric images from mapping platforms (hotel interiors, sports arenas, night views, lifestyle vlogs).

To optimize the dataset for reinforcement learning, we implemented careful processing: 
1) Removal of Google Map API watermarks containing address-related texts to prevent models from bypassing logical reasoning through OCR-based text detection;
2) Hierarchical labeling with (Country, Primary Administrative Division) format, where primary divisions are augmented with alias expansions (\eg, "Alba/Scotland" and "Sichuan/Sichuan Province") to address naming variations;
3) Difficulty stratification via joint predictions from two leading multimodal models (Kimi1.5 and Doubao1.5-Vision-Pro). Samples are categorized into Easy (correctly identified by both Kimi1.5 and Doubao1.5-Vision-Pro), Medium (recognized by only one model), and Hard (challenging for both models).

## Data from Xiaohongshu App

Considering that the data from Google Maps might have been collected and used for pre-training by large models, and as the Xiaohongshu App is an active community for sharing travel and lifestyle experiences, it contains the latest geographical images uploaded by users. We have constructed a new test dataset on the Xiaohongshu App. We retrieved the images of less-known street scenes in different provinces of China, which were uploaded by users around April 14, 2025. And through the Doubao model, we retained the images that do not directly contain geographical location information but imply some clues.  

## Data from openai o3 model
So far, through conversations with o3, we have collected 50 detailed image localization samples. Additionally, we have used SIFT matching to calculate the coordinates [x1, y1, x2, y2] of the cropped images from o3's thought process within the original images, where these values represent the normalized coordinates of the top-left and bottom-right corners respectively.
