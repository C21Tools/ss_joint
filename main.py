import cv2
import numpy as np
import streamlit as st
import datetime
from operator import attrgetter
from PIL import Image,ImageFont,ImageDraw
from zoneinfo import ZoneInfo


def main():
    st.title('C21アセン画像結合ツール')
    st.markdown('## 使い方')
    st.markdown('- 結合したいスクリーンショットをまとめてサイドバーにドラッグ＆ドロップすると、パーツツリーとステータスが結合された画像が生成されます。')
    st.markdown('- スクリーンショットの撮影方法は[Note](https://note.com/take_c21/n/nf7fc20d5558e)を参照してください。')
    st.markdown('- 解像度は800x600と1066x600（ワイド画面）に対応しています。')    
    st.markdown('- 結合後のスクリーンショットにメモや作成日時を入れることができます。')
    st.markdown('- スクリーンショットは画像処理のためサーバーにいったん送られますが、サーバーに保存されることはなく、ブラウザを閉じると消去されます。')
    uploaded_files = st.sidebar.file_uploader("↓にスクリーンショットをドラッグアンドドロップ", accept_multiple_files=True)

    memo_text = st.sidebar.text_area('テキストを入力')
    check_datetime = st.sidebar.checkbox('作成日時を入れる')
    if uploaded_files:    
        files_img=sorted(uploaded_files, key=attrgetter('name'))
        files_template=np.roll(files_img,-1)
        frame = Image.open(files_img[-1])
        if frame.width == 1066:
            frame = frame.crop((133,0,932,601))
        frame = np.array(frame)
        header = frame[71:90,18:211]
        footer = frame[90:408,18:211]
        result = header

        for file_img,file_template in zip(files_img[:-1],files_template[:-1]):
            img = Image.open(file_img)
            if img.width == 1066:
                img = img.crop((133,0,932,601))
            img = np.array(img)
            img = img[90:297,18:211]
            template = Image.open(file_template)
            if template.width == 1066:
                template = template.crop((133,0,932,601))
            template = np.array(template)
            template = template[90:100,18:211]
            match = cv2.matchTemplate(img,template,cv2.TM_CCOEFF_NORMED)
            minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(match)
            img = img[0:maxLoc[1],:]
            result = cv2.vconcat([result,img])

        result = cv2.vconcat([result,footer])

        status = frame[71:408,234:790]
        status = cv2.copyMakeBorder(status, 0, result.shape[0]-status.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(255,255,255)) 

        result = cv2.hconcat([result,status])

        result_memo = Image.fromarray(result)
        result_memo_draw = ImageDraw.Draw(result_memo)
        font = ImageFont.truetype(font="./fonts/NotoSansJP-Regular.otf", size=10)
        result_memo_draw.text((200,350),memo_text,(0,0,0),font=font)
        if check_datetime:
            dt_now = datetime.datetime.now(ZoneInfo("Asia/Tokyo"))
            dt_pos = [result_memo.width-170,result_memo.height-20]
            result_memo_draw.text(dt_pos,dt_now.strftime('%Y/%m/%d %H:%M'),(0,0,0), font= font)
            result_memo = np.array(result_memo)                                                         
        st.image(result_memo,output_format = 'png')
            
if __name__ == '__main__':
    main()
