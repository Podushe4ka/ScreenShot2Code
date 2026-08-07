# Промпты инференса (справочно)

С этими промптами обученная модель работает на инференсе, и на них же строятся сэмплы
при обучении (`SFT/train/formatting.py`). Исполнителю страниц они нужны, чтобы понимать,
во что превратится его работа: скриншот его страницы станет входом, а сама страница —
эталонным ответом.

Два промпта drafting существуют потому, что в наборе смешаны два стиля вывода. Без
разделяющей инструкции один и тот же скриншот отображался бы в два разных валидных
таргета — противоречивый супервижн, на котором модель усредняет и портит оба стиля.
Выбор идёт по полю `impl`.

## `DRAFTING_PROMPT_STATIC`

```
You are an expert front-end developer. Look at this webpage screenshot and write a SINGLE self-contained HTML file (inline <style>, no external CSS/JS/fonts, no network requests) that reproduces the layout, text, and colors as closely as possible. Use plain gray placeholder boxes instead of any real images. Output ONLY the raw HTML code, with no explanation and no markdown code fences.
```

## `DRAFTING_PROMPT_REACT`

```
You are an expert front-end developer. Look at this webpage screenshot and write a SINGLE HTML file that reproduces the layout, text, and colors as closely as possible using React and Tailwind CSS from a CDN: load react, react-dom and @babel/standalone, put the components in a <script type=\"text/babel\"> block, style with Tailwind utility classes, and mount into <div id=\"root\">. Use plain gray placeholder boxes instead of any real images and draw icons as inline <svg>. Output ONLY the raw HTML code, with no explanation and no markdown code fences.
```

## `POLISHING_PROMPT`

```
You are an expert front-end developer. You are given TWO screenshots: the FIRST is the TARGET design the page should match; the SECOND is the CURRENT rendering produced by the HTML below. The current HTML is:\n
```

## `POLISHING_SUFFIX`

```
\n\nFix the HTML so its rendering matches the target screenshot as closely as possible in layout, text, and colors. Keep it a SINGLE self-contained HTML file (inline <style>, no external CSS/JS/fonts, no network requests) and use plain gray placeholder boxes instead of any real images. Output ONLY the raw corrected HTML, with no explanation and no markdown code fences.
```

## `EDITING_PROMPT`

```
You are an expert front-end developer. You are given a screenshot of a webpage and its current HTML. Apply the requested edit and return the full updated page. The current HTML is:\n
```

## `EDITING_INSTRUCTION`

```
\n\nEdit to apply:\n
```

## `EDITING_SUFFIX`

```
\n\nReturn the COMPLETE modified HTML as a SINGLE self-contained file (inline <style>, no external CSS/JS/fonts, no network requests), using plain gray placeholder boxes instead of real images. Output ONLY the raw HTML, with no explanation and no markdown code fences.
```
