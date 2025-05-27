---
layout: default
title: SAE Feature Analysis
---

<div class="tab-nav">
    <button class="tab-button active" id="about-tab" data-tab="about">About</button>
    <button class="tab-button" id="demo-tab" data-tab="demo">Demo</button>
</div>

<div id="about-content" class="tab-content active">
    <div class="about-content">
        {% include_relative about.md %}
    </div>
</div>

<div id="demo-content" class="tab-content">
    {% include feature-viewer.html %}
</div> 