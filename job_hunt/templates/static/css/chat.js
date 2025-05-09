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

  function scrollToBottom() {
    const chatArea = document.getElementById('area');
    if (chatArea) {
      chatArea.scrollTo({ top: chatArea.scrollHeight, behavior: 'smooth' });
    }
  }
  
  


  document.addEventListener('DOMContentLoaded', function () {
    const form = document.getElementById('chat-form');
    if (!form) return;
  
    form.addEventListener('submit', async function (event) {
      event.preventDefault();
  
      const input = document.getElementById('question');
      const question = input.value.trim();
      if (!question) return;
      input.value = '';
  
      const chatArea = document.getElementById('area');
  
      // 質問を表示
      const userDiv = document.createElement('div');
      userDiv.classList.add('d-flex', 'justify-content-end', 'mb-2');
      userDiv.innerHTML = `
        <div class="chat-bubble question card text-left">
          <p class="chat-text">${question}</p>
        </div>
      `;
      chatArea.appendChild(userDiv);
      scrollToBottom();
  
      // 考え中... を表示（あとで削除する用にidをつける）
      const thinkingDiv = document.createElement('div');
      thinkingDiv.classList.add('d-flex', 'justify-content-start', 'mb-2');
      thinkingDiv.setAttribute('id', 'pending-response');
      thinkingDiv.innerHTML = `
        <div class="chat-bubble answer card">
          <p class="chat-text m-3">考え中...</p>
        </div>
      `;
      chatArea.appendChild(thinkingDiv);
      scrollToBottom();
  
      try {
        const response = await fetch(window.location.href, {
          method: 'POST',
          headers: {
            'X-CSRFToken': getCookie('csrftoken'),
            'Content-Type': 'application/x-www-form-urlencoded'
          },
          body: new URLSearchParams({
            question: question,
            include_user_info: document.getElementById('include_user_info').checked
          })
        });
  
        if (!response.ok) throw new Error(`HTTPエラー: ${response.status}`);
        const data = await response.json();
  
        // 考え中...を削除
        const pending = document.getElementById('pending-response');
        if (pending) pending.remove();
  
        // 回答を表示
        const aiDiv = document.createElement('div');
        aiDiv.classList.add('d-flex', 'justify-content-start', 'mb-2');
        aiDiv.innerHTML = `
          <div class="chat-bubble answer card">
            <p class="chat-text m-3">${data.answer || data.error || 'エラーが発生しました。'}</p>
          </div>
        `;
        chatArea.appendChild(aiDiv);
        scrollToBottom();
  
      } catch (err) {
        console.error("Fetchエラー:", err);
  
        const pending = document.getElementById('pending-response');
        if (pending) pending.remove();
  
        const aiDiv = document.createElement('div');
        aiDiv.classList.add('d-flex', 'justify-content-start', 'mb-2');
        aiDiv.innerHTML = `
          <div class="chat-bubble answer card">
            <p class="chat-text m-3">通信エラーが発生しました。</p>
          </div>
        `;
        chatArea.appendChild(aiDiv);
        scrollToBottom();
      }
    });
  });
  
  function getCookie(name) {
    let cookieValue = null;
    if (document.cookie && document.cookie !== '') {
      for (const cookie of document.cookie.split(';')) {
        const [key, value] = cookie.trim().split('=');
        if (key === name) {
          cookieValue = decodeURIComponent(value);
          break;
        }
      }
    }
    return cookieValue;
  }