import cv2
from ultralytics import YOLO
import time

# 1. Tayyor YOLOv8 modelini yuklab olamiz
model = YOLO("yolov8n.pt")  # 'n' - nano variant, tez ishlaydi

# 2. Veb-kamerani yoqamiz
cap = cv2.VideoCapture(0)

# Kadr o'lchamlarini biroz kattalashtirish mumkin
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)

print("Imtihon nazorati (Proctoring) boshlandi... Chiqish uchun 'q' tugmasini bosing.")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Kameradan tasvir olishda xatolik!")
        break

    # Tasvirni gorizontaliga aylantiramiz (ko'zgudek ko'rinishi uchun)
    frame = cv2.flip(frame, 1)

    # 3. AI orqali kadrda obyektlarni qidiramiz
    results = model(frame, stream=True, verbose=False)

    phone_detected = False
    book_detected = False
    person_count = 0

    for r in results:
        boxes = r.boxes
        for box in boxes:
            cls_id = int(box.cls[0])
            class_name = model.names[cls_id]
            conf = float(box.conf[0])

            # Faqat ishonchliligi 0.4 dan yuqori bo'lganlarini olamiz (yolg'on xabarlarni kamaytirish uchun)
            if conf > 0.4:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Talabani aniqlash
                if class_name == "person":
                    person_count += 1
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(frame, "TALABA", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                # Telefonni aniqlash
                elif class_name == "cell phone":
                    phone_detected = True
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                    cv2.putText(frame, "TELEFON", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                # Kitob yoki daftarni aniqlash
                elif class_name == "book":
                    book_detected = True
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 3) # Apelsin rang
                    cv2.putText(frame, "KITOB/SHPARGALKA", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

    # 4. Qoidalarni tekshirish va xatoliklarni aniqlash
    warning_text = ""
    status_color = (0, 255, 0) # Yashil (Normal)
    
    if phone_detected:
        warning_text = "QOIDABUZARLIK: Telefon aniqlandi!"
        status_color = (0, 0, 255) # Qizil
    elif book_detected:
        warning_text = "QOIDABUZARLIK: Kitob/Daftar aniqlandi!"
        status_color = (0, 165, 255) # Apelsin
    elif person_count > 1:
        warning_text = f"QOIDABUZARLIK: Kadrdan {person_count} kishi topildi!"
        status_color = (0, 0, 255) # Qizil
    elif person_count == 0:
        warning_text = "OGOHLANTIRISH: Talaba joyida emas!"
        status_color = (0, 255, 255) # Sariq

    # Ekranga ma'lumotlarni chiqarish
    if warning_text:
        # Yuqori qismga qoidabuzarlik banneri
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 60), status_color, -1)
        cv2.putText(frame, warning_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Qat'iy qoidabuzarlik bo'lsa ekranni qizartirish va markazga yozuv qo'yish
        if phone_detected or person_count > 1:
            # Yarim shaffof qizil fon qo'shish
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 60), (frame.shape[1], frame.shape[0]), (0, 0, 255), -1)
            cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)
            
            cv2.putText(frame, "IMTIHON BLOKLANISHI MUMKIN!", (50, frame.shape[0] // 2), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
    else:
        # Normal holatda qora fon ustida yashil yozuv
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 40), (0, 0, 0), -1)
        cv2.putText(frame, "Holat: Normal (Qoidabuzarlik yo'q)", (20, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

    # Vaqt va kadr ma'lumotlarini pastki qismga qo'shish
    current_time = time.strftime("%H:%M:%S")
    cv2.rectangle(frame, (0, frame.shape[0] - 40), (frame.shape[1], frame.shape[0]), (0, 0, 0), -1)
    cv2.putText(frame, f"Vaqt: {current_time} | Odamlar soni: {person_count}", (10, frame.shape[0] - 15), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # 5. Natijani ekranda ko'rsatish
    cv2.imshow("DTM Imtihon Proctoring Tizimi", frame)

    # 'q' tugmasi bosilsa dastur to'xtaydi
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Kamerani o'chirish va oynalarni yopish
cap.release()
cv2.destroyAllWindows()