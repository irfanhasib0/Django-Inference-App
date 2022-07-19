var video_hls = document.getElementById('video_hls');
  if(Hls.isSupported()) {
    console.log('if --- 1')
    var hls = new Hls();
    hls.loadSource('http://localhost:9007/hls/stream.m3u8');
    hls.attachMedia(video_hls);
    hls.on(Hls.Events.MANIFEST_PARSED,function() {
      video_hls.play();
  });
 }
 // hls.js is not supported on platforms that do not have Media Source Extensions (MSE) enabled.
 // When the browser has built-in HLS support (check using `canPlayType`), we can provide an HLS manifest (i.e. .m3u8 URL) directly to the video element throught the `src` property.
 // This is using the built-in support of the plain video element, without using hls.js.
 // Note: it would be more normal to wait on the 'canplay' event below however on Safari (where you are most likely to find built-in HLS support) the video.src URL must be on the user-driven
 // white-list before a 'canplay' event will be emitted; the last video event that can be reliably listened-for when the URL is not on the white-list is 'loadedmetadata'.
  else if (video.canPlayType('application/vnd.apple.mpegurl')) {
    console.log('if --- 2')
    video_hls.src = 'http://localhost:9007/hls/stream.m3u8';
    video_hls.addEventListener('loadedmetadata',function() {
    video_hls.play();
    });
  }
