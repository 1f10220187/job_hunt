window.onload = function() {
    // ↪︎ windowがロードされた時にアクションを実行するように設定
      if (document.getElementById("area")) {
        // ↪︎ areaのIDがある場合に処理を実行させる（これがないとチャット画面がなくても常にJavaScriptが動いてしまいます）
        var scrollPosition = document.getElementById("area").scrollTop;
        // ↪︎ area要素のスクロールされた時の最も高い場所を取得
        var scrollHeight = document.getElementById("area").scrollHeight;
        // ↪︎ area要素自体の最も高い場所を取得
        document.getElementById("area").scrollTop = scrollHeight;
        // ↪︎ area要素のスクロールされた時の最も高い場所をarea要素自体の最も高い場所として指定してあげる
      }
    }
    
document.addEventListener('DOMContentLoaded', function() {
      // ロード時に保存されたタブの状態を取得
      const activeTab = localStorage.getItem('activeTab');
      if (activeTab) {
          const activeTabElement = document.querySelector(activeTab);
          if (activeTabElement) {
              activeTabElement.click();
          }
      }
  
      // タブが変更された時に選択されたタブを保存
      const tabs = document.querySelectorAll('.nav-link');
      tabs.forEach(tab => {
          tab.addEventListener('click', function() {
              localStorage.setItem('activeTab', `#${tab.getAttribute('aria-controls')}`);
          });
      });
  });
  