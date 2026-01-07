# -*- coding: utf-8 -*-
"""
認証コンポーネント
シンプルなパスワード認証を提供
"""
import streamlit as st
import hashlib

# デフォルト認証情報（本番環境ではsecrets.tomlを使用）
DEFAULT_CREDENTIALS = {
    "admin": "admin123",
    "user": "user123"
}


def hash_password(password: str) -> str:
    """パスワードをハッシュ化"""
    return hashlib.sha256(password.encode()).hexdigest()


def check_authentication() -> bool:
    """認証状態をチェック"""
    return st.session_state.get("authenticated", False)


def get_credentials():
    """認証情報を取得（secrets.tomlまたはデフォルト）"""
    try:
        # secrets.tomlから読み込み
        credentials = {}
        if hasattr(st, 'secrets') and 'passwords' in st.secrets:
            for username in st.secrets.get('credentials', {}).get('usernames', []):
                if username in st.secrets['passwords']:
                    credentials[username] = st.secrets['passwords'][username]
        if credentials:
            return credentials
    except Exception:
        pass
    return DEFAULT_CREDENTIALS


def show_login_form():
    """ログインフォームを表示"""
    st.markdown("""
    <style>
        .login-container {
            max-width: 400px;
            margin: 0 auto;
            padding: 2rem;
            background-color: #f8f9fa;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
    </style>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        st.markdown("### 🔐 ログイン")

        with st.form("login_form"):
            username = st.text_input("ユーザー名", placeholder="ユーザー名を入力")
            password = st.text_input("パスワード", type="password", placeholder="パスワードを入力")
            submit = st.form_submit_button("ログイン", use_container_width=True)

            if submit:
                credentials = get_credentials()
                if username in credentials and credentials[username] == password:
                    st.session_state["authenticated"] = True
                    st.session_state["username"] = username
                    st.rerun()
                else:
                    st.error("ユーザー名またはパスワードが正しくありません")

        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; color: #666; font-size: 0.8rem;">
            <p>デモ用アカウント</p>
            <p>ユーザー名: admin / パスワード: admin123</p>
        </div>
        """, unsafe_allow_html=True)


def show_logout_button():
    """ログアウトボタンを表示"""
    if st.sidebar.button("🚪 ログアウト", use_container_width=True):
        st.session_state["authenticated"] = False
        st.session_state["username"] = None
        st.rerun()


def require_authentication(func):
    """認証が必要なページ用デコレータ"""
    def wrapper(*args, **kwargs):
        if not check_authentication():
            show_login_form()
            st.stop()
        return func(*args, **kwargs)
    return wrapper
