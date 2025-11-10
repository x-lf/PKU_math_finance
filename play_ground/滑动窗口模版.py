class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        st = set([])

        arr_lens = len(s)

        i = 0
        j = 0

        res = 0

        while(j<arr_lens):
            if not (s[j] in st):
                # print('step',i,j)
                st.add(s[j])
                
                # print(f'add s[{j}]:{s[j]}')
                # print(st)
            else:
                while((s[j] in st) and (i<j)):
                    # print('step',i,j)
                    # print(st)
                    # print(f'remove s{[i]}:{s[i]}')
                    st.remove(s[i])
                    i = i+1
                st.add(s[j])


            if j-i+1 > res:
                res = j-i+1
            
            j = j+1
        
        return res
            

     

        