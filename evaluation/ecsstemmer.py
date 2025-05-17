import re

class EcsStemmer:
    def __init__(self):
        self.rootWords = open("rootwords.txt").read()

    def stemm(self, words):
        result = []
        for word in words:
            if self.isRootWord(word): result.append(word)
            else: result.append(self.removingProcess(word))
        return result

    def stemmWord(self, word):
        #if self.isRootWord(word): result.append(word)
        #else: result.append(self.removingProcess(word))
        if self.isRootWord(word): return word
        else: return self.removingProcess(word)
        
    def removingProcess(self, word):
        return self.removeDerivationPrefixes(self.removeDerivationSuffixes(self.removeInflectionSuffixes(word)))


    def isRootWord(self, word):
        
        escaped_word = re.escape(word)
        if re.search(escaped_word + " ", self.rootWords):
            return 1
        else:
            return 0

    # /* check the Rule Precedence
    # * combination of Prefix and Suffix
    # * "be-lah" "be-an" "me-i" "di-i" "pe-i" or "te-i"
    # */
    def isruleprecedence(self, word):
        if re.search(r"^(be)([A-Za-z\-]+)(lah|an)$", word): return 1
        elif re.search(r"^(di|[mpt]e)([A-Za-z\-]+)(i)$", word): return 1
        else: return 0

    # /* check Disallowed Prefix-Suffix Combinations
    # * "be-i" . "di-an" . "ke-i|kan" . "me-an" . "se-i|kan" or "te-an"
    # * **/
    def isdisallowedprefixsuffixes(self, word):
        if re.search(r"^(be)([A-Za-z\-]+)(i)$", word): return 1
        elif re.search(r"^(di)([A-Za-z\-]+)(an)$", word): return 1
        elif re.search(r"^(ke)([A-Za-z\-]+)(i|kan)$", word): return 1
        elif re.search(r"^(me)([A-Za-z\-]+)(an)$", word): return 1
        elif re.search(r"^(se)([A-Za-z\-]+)(i|kan)$", word): return 1
        elif re.search(r"^(te)([A-Za-z\-]+)(an)$", word): return 1
        else: return 0

    # /* remove Inflection Suffixes, 
    # * 1. Particle "-lah" "-kah" "-tah" and "-pun" 
    # * 2. Possesive Pronoun "-ku" "-mu" "-nya"
    # */
    def removeInflectionSuffixes(self, word):
        baseWord = word
        if re.search(r"([klt]ah|pun|[km]u|nya)$", word):
            infSuf = re.sub(r"([klt]ah|pun|[km]u|nya)$", "", word)
            if re.search(r"([klt]ah|pun)$", word):
                if re.search(r"([km]u|nya)$", word):
                    posPron = re.sub(r"([km]u|nya)$", "", infSuf)
                    return posPron
            return infSuf
        else: return baseWord

    # /* remove Dervation Suffixes
    # * "-i" . "-kan" . "-an"
    # * **/
    def removeDerivationSuffixes(self, word):
        baseWord = ""

        if re.search(r"(kan)$", word):
            baseWord = re.sub(r"(kan)$", "", word)
            if self.isRootWord(baseWord): return baseWord

        elif re.search(r"(an|i)$", word):
            baseWord = re.sub(r"(an|i)$", "", word)
            if self.isRootWord(baseWord): return baseWord

        if self.isdisallowedprefixsuffixes(baseWord): return word

        return word

    # /* remove Derivation Prefixes
    # * "di-" . "ke-" . "se-" . "me-" . "be-" . "pe-" or "te-"
    # * */
    def removeDerivationPrefixes(self, word):
        baseWord = word
        strRemovedDerivSuff = ""
        strRemovedStdPref = ""
        strRemovedCmplxPref = ""

        # /************** check Standard Prefix "di-" . "ke-" . "se-" **************/
        if re.search(r"^(di|[ks]e)\S{1,}", word):
            strRemovedStdPref = re.sub(r"^(di|[ks]e)", "", word)
            if self.isRootWord(strRemovedStdPref): return strRemovedStdPref

            strRemovedDerivSuff = self.removeDerivationSuffixes(strRemovedStdPref)
            if self.isRootWord(strRemovedDerivSuff): return strRemovedDerivSuff

        # /* rule 35 */
        elif re.search(r"^([^aiueo])e\1[aiueo]\S{1,}", word):
            prefixes = re.sub(r"^([^aiueo])e", "", word)
            if self.isRootWord(prefixes): return prefixes

            strRemovedDerivSuff = self.removeDerivationSuffixes(prefixes)
            if self.isRootWord(strRemovedDerivSuff): return strRemovedDerivSuff

        # /************** check Complex Prefixes "te-"."me-"."be-"."pe-" **************/
        elif re.search(r"^([tmbp]e)\S{1,}", word):
            # /************** Prefix "be-" **************/
            if re.search(r"^(be)\S{1,}", word):
                # /* if prefix "be-" */
                if re.search(r"^(ber)[aiueo]\S{1,}", word):
                    # /* rule 1 */
                    strRemovedCmplxPref = re.sub(r"^(ber)", "", word)
                    if self.isRootWord(strRemovedCmplxPref): return strRemovedCmplxPref

                    strRemovedDerivSuff = self.removeDerivationSuffixes(strRemovedCmplxPref)
                    if self.isRootWord(strRemovedDerivSuff): return strRemovedDerivSuff

                    strRemovedCmplxPref = re.sub(r"^(ber)", "r", word)
                    if self.isRootWord(strRemovedCmplxPref): return strRemovedCmplxPref

                    strRemovedDerivSuff = self.removeDerivationSuffixes(strRemovedCmplxPref)
                    if self.isRootWord(strRemovedDerivSuff): return strRemovedDerivSuff

                elif re.search(r"^(ber)[^aiueor]([A-Za-z\-]+)(?!er)\S{1,}", word):
                    # /* rule 2 */
                    strRemovedCmplxPref = re.sub(r"^(ber)", "", word)
                    if self.isRootWord(strRemovedCmplxPref): return strRemovedCmplxPref

                    strRemovedDerivSuff = self.removeDerivationSuffixes(strRemovedCmplxPref)
                    if self.isRootWord(strRemovedDerivSuff): return strRemovedDerivSuff

                elif re.search(r"^(ber)[^aiueor]([A-Za-z\-]+)er[aiueo]\S{1,}", word):
                    # /* rule 3 */
                    strRemovedCmplxPref = re.sub(r"^(ber)", "", word)
                    if self.isRootWord(strRemovedCmplxPref): return strRemovedCmplxPref

                    strRemovedDerivSuff = self.removeDerivationSuffixes(strRemovedCmplxPref)
                    if self.isRootWord(strRemovedDerivSuff): return strRemovedDerivSuff

                elif re.search(r"^belajar\S{0,}", word):
                    # /* rule 4 */
                    strRemovedCmplxPref = re.sub(r"^(bel)", "", word)
                    if self.isRootWord(strRemovedCmplxPref): return strRemovedCmplxPref

                    strRemovedDerivSuff = self.removeDerivationSuffixes(strRemovedCmplxPref)
                    if self.isRootWord(strRemovedDerivSuff): return strRemovedDerivSuff

                elif re.search(r"^(be)[^aiueolr]er[^aiueo]\S{1,}", word):
                    # /* rule 5 */
                    strRemovedCmplxPref = re.sub(r"^(be)", "", word)
                    if self.isRootWord(strRemovedCmplxPref): return strRemovedCmplxPref

                    strRemovedDerivSuff = self.removeDerivationSuffixes(strRemovedCmplxPref)
                    if self.isRootWord(strRemovedDerivSuff): return strRemovedDerivSuff
            # /************** end Prefix "be-" **************/

            # /************** Prefix "te-" **************/
            if re.search(r"^(te)\S{1,}", word):
                if re.search(r"^(terr)\S{1,}", word): return word

                elif re.search(r"^(ter)[aiueo]\S{1,}", word):
                    # /* rule 6 */
                    strRemovedCmplxPref = re.sub(r"^(ter)", "", word)
                    if self.isRootWord(strRemovedCmplxPref): return strRemovedCmplxPref

                    strRemovedDerivSuff = self.removeDerivationSuffixes(strRemovedCmplxPref)
                    if self.isRootWord(strRemovedCmplxPref): return strRemovedDerivSuf

                    strRemovedCmplxPref = re.sub(r"^(ter)", "r", word)
                    if self.isRootWord(strRemovedCmplxPref): return strRemovedCmplxPre

                    strRemovedDerivSuff = self.removeDerivationSuffixes(strRemovedCmplxPref)
                    if self.isRootWord(strRemovedDerivSuff): return strRemovedDerivSuff

                elif re.search(r"^(ter)[^aiueor]er[aiueo]\S{1,}", word):
                    # /* rule 7 */
                    strRemovedCmplxPref = re.sub(r"^(ter)", "", word)
                    if self.isRootWord(strRemovedCmplxPref): return strRemovedCmplxPref

                    strRemovedDerivSuff = self.removeDerivationSuffixes(strRemovedCmplxPref)
                    if self.isRootWord(strRemovedDerivSuff): return strRemovedDerivSuff

                elif re.search(r"^(ter)[^aiueor](?!er)\S{1,}", word):
                    # /* rule 8 */
                    strRemovedCmplxPref = re.sub(r"^(ter)", "", word)
                    if self.isRootWord(strRemovedCmplxPref): return strRemovedCmplxPref

                    strRemovedDerivSuff = self.removeDerivationSuffixes(strRemovedCmplxPref)
                    if self.isRootWord(strRemovedDerivSuff): return strRemovedDerivSuff

                elif re.search(r"^(te)[^aiueor]er[aiueo]\S{1,}", word):
                    # /* rule 9 */
                    strRemovedCmplxPref = re.sub(r"^(te)", "", word)
                    if self.isRootWord(strRemovedCmplxPref): return strRemovedCmplxPref

                    strRemovedDerivSuff = self.removeDerivationSuffixes(strRemovedCmplxPref)
                    if self.isRootWord(strRemovedDerivSuff): return strRemovedDerivSuff

                elif re.search(r"^(ter)[^aiueor]er[^aiueo]\S{1,}", word):
                    # /* rule 35 */
                    strRemovedCmplxPref = re.sub(r"^(ter)", "", word)
                    if self.isRootWord(strRemovedCmplxPref): return strRemovedCmplxPref

                    strRemovedDerivSuff = self.removeDerivationSuffixes(strRemovedCmplxPref)
                    if self.isRootWord(strRemovedDerivSuff): return strRemovedDerivSuff
            # /************** end Prefix "te-" **************/

            # /************** Prefix "me-" **************/
            if re.search(r"^(me)\S{1,}",word):
                if re.search(r"^(me)[lrwyv][aiueo]",word):
                    # /* rule 10 */
                    strRemovedCmplxPref = re.sub(r"^(me)", "", word)
                    if self.isRootWord(strRemovedCmplxPref): return strRemovedCmplxPref

                    strRemovedDerivSuff = self.removeDerivationSuffixes(strRemovedCmplxPref)
                    if self.isRootWord(strRemovedDerivSuff): return strRemovedDerivSuff

                elif re.search(r"^(mem)[bfvp]\S{1,}",word):
                    # /* rule 11 */
                    strRemovedCmplxPref = re.sub(r"^(mem)", "", word)
                    if self.isRootWord(strRemovedCmplxPref): return strRemovedCmplxPref

                    strRemovedDerivSuff = self.removeDerivationSuffixes(strRemovedCmplxPref)
                    if self.isRootWord(strRemovedDerivSuff): return strRemovedDerivSuff

                elif re.search(r"^(mem)((r[aiueo])|[aiueo])\S{1,}", word):
                    # /* rule 13 */
                    strRemovedCmplxPref = re.sub(r"^(mem)", "m", word)
                    if self.isRootWord(strRemovedCmplxPref): return strRemovedCmplxPref

                    strRemovedDerivSuff = self.removeDerivationSuffixes(strRemovedCmplxPref)
                    if self.isRootWord(strRemovedDerivSuff): return strRemovedDerivSuff

                    strRemovedCmplxPref = re.sub(r"^(mem)", "p", word)
                    if self.isRootWord(strRemovedCmplxPref): return strRemovedCmplxPref

                    strRemovedDerivSuff = self.removeDerivationSuffixes(strRemovedCmplxPref)
                    if self.isRootWord(strRemovedDerivSuff): return strRemovedDerivSuff

                elif re.search(r"^(men)[cdjszt]\S{1,}", word):
                    # /* rule 14 */
                    strRemovedCmplxPref = re.sub(r"^(men)", "", word)
                    if self.isRootWord(strRemovedCmplxPref): return strRemovedCmplxPref

                    strRemovedDerivSuff = self.removeDerivationSuffixes(strRemovedCmplxPref)
                    if self.isRootWord(strRemovedDerivSuff): return strRemovedDerivSuff

                elif re.search(r"^(men)[aiueo]\S{1,}", word):
                    # /* rule 15 */
                    strRemovedCmplxPref = re.sub(r"^(men)", "n", word)
                    if self.isRootWord(strRemovedCmplxPref): return strRemovedCmplxPref

                    strRemovedDerivSuff = self.removeDerivationSuffixes(strRemovedCmplxPref)
                    if self.isRootWord(strRemovedDerivSuff): return strRemovedDerivSuff

                    strRemovedCmplxPref = re.sub(r"^(men)", "t", word)
                    if self.isRootWord(strRemovedCmplxPref): return strRemovedCmplxPref

                    strRemovedDerivSuff = self.removeDerivationSuffixes(strRemovedCmplxPref)
                    if self.isRootWord(strRemovedDerivSuff): return strRemovedDerivSuff

                elif re.search(r"^(meng)[ghqk]\S{1,}", word):
                   # /* rule 16 */
                    strRemovedCmplxPref = re.sub(r"^(meng)", "", word)
                    if self.isRootWord(strRemovedCmplxPref): return strRemovedCmplxPref

                    strRemovedDerivSuff = self.removeDerivationSuffixes(strRemovedCmplxPref)
                    if self.isRootWord(strRemovedDerivSuff): return strRemovedDerivSuff

                elif re.search(r"^(meng)[aiueo]\S{1,}", word):
                   # /* rule 17 */
                    strRemovedCmplxPref = re.sub(r"^(meng)", "", word)
                    if self.isRootWord(strRemovedCmplxPref): return strRemovedCmplxPref

                    strRemovedDerivSuff = self.removeDerivationSuffixes(strRemovedCmplxPref)
                    if self.isRootWord(strRemovedDerivSuff): return strRemovedDerivSuff

                    strRemovedCmplxPref = re.sub(r"^(meng)", "k", word)
                    if self.isRootWord(strRemovedCmplxPref): return strRemovedCmplxPref

                    strRemovedDerivSuff = self.removeDerivationSuffixes(strRemovedCmplxPref)
                    if self.isRootWord(strRemovedDerivSuff): return strRemovedDerivSuff

                    strRemovedCmplxPref = re.sub(r"^(menge)", "", word)
                    if self.isRootWord(strRemovedCmplxPref): return strRemovedCmplxPref

                    strRemovedDerivSuff = self.removeDerivationSuffixes(strRemovedCmplxPref)
                    if self.isRootWord(strRemovedDerivSuff): return strRemovedDerivSuff

                elif re.search(r"^(meny)[aiueo]\S{1,}", word):
                    # /* rule 18 */
                    strRemovedCmplxPref = re.sub(r"^(meny)", "s", word)
                    if self.isRootWord(strRemovedCmplxPref): return strRemovedCmplxPref

                    strRemovedDerivSuff = self.removeDerivationSuffixes(strRemovedCmplxPref)
                    if self.isRootWord(strRemovedDerivSuff): return strRemovedDerivSuff

                    strRemovedCmplxPref = re.sub(r"^(me)", "", word)
                    if self.isRootWord(strRemovedCmplxPref): return strRemovedCmplxPref

                    strRemovedDerivSuff = self.removeDerivationSuffixes(strRemovedCmplxPref)
                    if self.isRootWord(strRemovedDerivSuff): return strRemovedDerivSuff
            # /************** end Prefix "me-" **************/

            # /************** Prefix "pe-" **************/
            if re.search(r"^(pe)\S{1,}", word):
                if re.search(r"^(pe)[wy]\S{1,}", word):
                   # /* rule 20 */
                    strRemovedCmplxPref = re.sub(r"^(pe)", "", word)
                    if self.isRootWord(strRemovedCmplxPref): return strRemovedCmplxPref

                    strRemovedDerivSuff = self.removeDerivationSuffixes(strRemovedCmplxPref)
                    if self.isRootWord(strRemovedDerivSuff): return strRemovedDerivSuff

                elif re.search(r"^(per)[aiueo]\S{1,}", word):
                    # /* rule 21 */
                    strRemovedCmplxPref = re.sub(r"^(per)", "", word)
                    if self.isRootWord(strRemovedCmplxPref): return strRemovedCmplxPref

                    strRemovedDerivSuff = self.removeDerivationSuffixes(strRemovedCmplxPref)
                    if self.isRootWord(strRemovedDerivSuff): return strRemovedDerivSuff

                    strRemovedCmplxPref = re.sub(r"^(per)", "r", word)
                    if self.isRootWord(strRemovedCmplxPref): return strRemovedCmplxPref

                    strRemovedDerivSuff = self.removeDerivationSuffixes(strRemovedCmplxPref)
                    if self.isRootWord(strRemovedDerivSuff): return strRemovedDerivSuff

                elif re.search(r"^(per)[^aiueor]([A-Za-z\-]+)(?!er)\S{1,}", word):
                    # /* rule 23 */
                    strRemovedCmplxPref = re.sub(r"^(per)", "", word)
                    if self.isRootWord(strRemovedCmplxPref): return strRemovedCmplxPref

                    strRemovedDerivSuff = self.removeDerivationSuffixes(strRemovedCmplxPref)
                    if self.isRootWord(strRemovedDerivSuff): return strRemovedDerivSuff

                elif re.search(r"^(per)[^aiueo]([A-Za-z\-]+)(er)[aiueo]\S{1,}", word):
                    # /* rule 24 */
                    strRemovedCmplxPref = re.sub(r"^(per)", "", word)
                    if self.isRootWord(strRemovedCmplxPref): return strRemovedCmplxPref

                    strRemovedDerivSuff = self.removeDerivationSuffixes(strRemovedCmplxPref)
                    if self.isRootWord(strRemovedDerivSuff): return strRemovedDerivSuff

                elif re.search(r"^(pem)[bfv]\S{1,}", word):
                    # /* rule 25 */
                    strRemovedCmplxPref = re.sub(r"^(pem)", "", word)
                    if self.isRootWord(strRemovedCmplxPref): return strRemovedCmplxPref

                    strRemovedDerivSuff = self.removeDerivationSuffixes(strRemovedCmplxPref)
                    if self.isRootWord(strRemovedDerivSuff): return strRemovedDerivSuff

                elif re.search(r"^(pem)(r[aiueo]|[aiueo])\S{1,}", word):
                    # /* rule 26 */
                    strRemovedCmplxPref = re.sub(r"^(pem)", "m", word)
                    if self.isRootWord(strRemovedCmplxPref): return strRemovedCmplxPref

                    strRemovedDerivSuff = self.removeDerivationSuffixes(strRemovedCmplxPref)
                    if self.isRootWord(strRemovedDerivSuff): return strRemovedDerivSuff

                    strRemovedCmplxPref = re.sub(r"^(pem)", "p", word)
                    if self.isRootWord(strRemovedCmplxPref): return strRemovedCmplxPref

                    strRemovedDerivSuff = self.removeDerivationSuffixes(strRemovedCmplxPref)
                    if self.isRootWord(strRemovedDerivSuff): return strRemovedDerivSuff

                elif re.search(r"^(pen)[cdjzt]\S{1,}", word):
                    # /* rule 27 */
                    strRemovedCmplxPref = re.sub(r"^(pen)", "", word)
                    if self.isRootWord(strRemovedCmplxPref): return strRemovedCmplxPref

                    strRemovedDerivSuff = self.removeDerivationSuffixes(strRemovedCmplxPref)
                    if self.isRootWord(strRemovedDerivSuff): return strRemovedDerivSuff

                elif re.search(r"^(pen)[aiueo]\S{1,}", word):
                    # /* rule 28 */
                    strRemovedCmplxPref = re.sub(r"^(pen)", "n", word)
                    if self.isRootWord(strRemovedCmplxPref): return strRemovedCmplxPref

                    strRemovedDerivSuff = self.removeDerivationSuffixes(strRemovedCmplxPref)
                    if self.isRootWord(strRemovedDerivSuff): return strRemovedDerivSuff

                    strRemovedCmplxPref = re.sub(r"^(pen)", "t", word)
                    if self.isRootWord(strRemovedCmplxPref): return strRemovedCmplxPref

                    strRemovedDerivSuff = self.removeDerivationSuffixes(strRemovedCmplxPref)
                    if self.isRootWord(strRemovedDerivSuff): return strRemovedDerivSuff

                elif re.search(r"^(peng)[^ghq]\S{1,}", word):
                    # /* rule 29 */
                    strRemovedCmplxPref = re.sub(r"^(peng)", "", word)
                    if self.isRootWord(strRemovedCmplxPref): return strRemovedCmplxPref

                    strRemovedDerivSuff = self.removeDerivationSuffixes(strRemovedCmplxPref)
                    if self.isRootWord(strRemovedDerivSuff): return strRemovedDerivSuff

                elif re.search(r"^(peng)[aiueo]\S{1,}", word):
                    # /* rule 30 */
                    strRemovedCmplxPref = re.sub(r"^(peng)", "", word)
                    if self.isRootWord(strRemovedCmplxPref): return strRemovedCmplxPref

                    strRemovedDerivSuff = self.removeDerivationSuffixes(strRemovedCmplxPref)
                    if self.isRootWord(strRemovedDerivSuff): return strRemovedDerivSuff

                    strRemovedCmplxPref = re.sub(r"^(peng)", "k", word)
                    if self.isRootWord(strRemovedCmplxPref): return strRemovedCmplxPref

                    strRemovedDerivSuff = self.removeDerivationSuffixes(strRemovedCmplxPref)
                    if self.isRootWord(strRemovedDerivSuff): return strRemovedDerivSuff

                    strRemovedCmplxPref = re.sub(r"^(penge)", "", word)
                    if self.isRootWord(strRemovedCmplxPref): return strRemovedCmplxPref

                    strRemovedDerivSuff = self.removeDerivationSuffixes(strRemovedCmplxPref)
                    if self.isRootWord(strRemovedDerivSuff): return strRemovedDerivSuff

                elif re.search(r"^(peny)[aiueo]\S{1,}", word):
                    # /* rule 31 */
                    strRemovedCmplxPref = re.sub(r"^(peny)", "s", word)
                    if self.isRootWord(strRemovedCmplxPref): return strRemovedCmplxPref

                    strRemovedDerivSuff = self.removeDerivationSuffixes(strRemovedCmplxPref)
                    if self.isRootWord(strRemovedDerivSuff): return strRemovedDerivSuff

                    strRemovedCmplxPref = re.sub(r"^(pe)", "", word)
                    if self.isRootWord(strRemovedCmplxPref): return strRemovedCmplxPref

                    strRemovedDerivSuff = self.removeDerivationSuffixes(strRemovedCmplxPref)
                    if self.isRootWord(strRemovedDerivSuff): return strRemovedDerivSuff

                elif re.search(r"^(pel)[aiueo]\S{1,}", word):
                    # /* rule 32 */
                    strRemovedCmplxPref = re.sub(r"^(pel)", "l", word)
                    if self.isRootWord(strRemovedCmplxPref): return strRemovedCmplxPref

                    strRemovedDerivSuff = self.removeDerivationSuffixes(strRemovedCmplxPref)
                    if self.isRootWord(strRemovedDerivSuff): return strRemovedDerivSuff

                elif re.search(r"^(pelajar)\S{0,}", word):
                    strRemovedCmplxPref = re.sub(r"^(pel)", "", word)
                    if self.isRootWord(strRemovedCmplxPref): return strRemovedCmplxPref

                    strRemovedDerivSuff = self.removeDerivationSuffixes(strRemovedCmplxPref)
                    if self.isRootWord(strRemovedDerivSuff): return strRemovedDerivSuff

                elif re.search(r"^(pe)[^rwylmn]er[aiueo]\S{1,}", word):
                    # /* rule 33 */
                    strRemovedCmplxPref = re.sub(r"^(pe)", "", word)
                    if self.isRootWord(strRemovedCmplxPref): return strRemovedCmplxPref

                    strRemovedDerivSuff = self.removeDerivationSuffixes(strRemovedCmplxPref)
                    if self.isRootWord(strRemovedDerivSuff): return strRemovedDerivSuff

                elif re.search(r"^(pe)[^rwylmn](?!er)\S{1,}", word):
                    # /* rule 34 */
                    strRemovedCmplxPref = re.sub(r"^(pe)", "", word)
                    if self.isRootWord(strRemovedCmplxPref): return strRemovedCmplxPref

                    strRemovedDerivSuff = self.removeDerivationSuffixes(strRemovedCmplxPref)
                    if self.isRootWord(strRemovedDerivSuff): return strRemovedDerivSuff

                elif re.search(r"^(pe)[^aiueor]er[^aiueo]\S{1,}", word):
                    # /* rule 36 */
                    strRemovedCmplxPref = re.sub(r"^(pe)", "", word)
                    if self.isRootWord(strRemovedCmplxPref): return strRemovedCmplxPref

                    strRemovedDerivSuff = self.removeDerivationSuffixes(strRemovedCmplxPref)
                    if self.isRootWord(strRemovedDerivSuff): return strRemovedDerivSuff
            # /************** end Prefix "pe-" **************/

        return baseWord