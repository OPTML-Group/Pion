document.addEventListener("DOMContentLoaded", function() {
  const roadmapImg = document.getElementById('roadmap-video');
  if (roadmapImg) {
    const gifUrl = roadmapImg.getAttribute('data-gif');
    if (gifUrl) {
      const img = new Image();
      img.onload = function() {
        roadmapImg.src = gifUrl;
      };
      img.src = gifUrl;
    }
  }
});

